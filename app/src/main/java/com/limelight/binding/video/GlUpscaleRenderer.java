package com.limelight.binding.video;

import android.content.res.AssetManager;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLExt;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.Matrix;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.view.Surface;

import com.limelight.LimeLog;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * GFN fork: OpenGL ES post-processing for the video stream.
 *
 * MediaCodec decodes into a SurfaceTexture owned by this class. Every new frame is
 * then drawn into the SurfaceView with up to three passes:
 *   1. External OES texture -> RGBA texture at stream resolution
 *   2. AMD FSR 1 EASU upscaling to the output resolution (only when upscaling)
 *   3. AMD FSR 1 RCAS sharpening into the window surface
 * or, with MWSR, two passes:
 *   1. External OES texture -> RGBA texture at stream resolution (first row = top)
 *   2. MWSR upscale or reduction straight into the window surface
 *
 * FSR 1 is Copyright (c) 2021 Advanced Micro Devices, Inc. (MIT license).
 * MWSR is Copyright (c) 2026 Bruno Martin (MIT license, see assets/mwsr/LICENSE-MWSR).
 */
public class GlUpscaleRenderer implements SurfaceTexture.OnFrameAvailableListener {
    private static final String COPY_VERTEX_SHADER =
            "attribute vec4 aPosition;\n" +
            "attribute vec4 aTexCoord;\n" +
            "uniform mat4 uTexMatrix;\n" +
            "varying vec2 vTexCoord;\n" +
            "void main() {\n" +
            "    gl_Position = aPosition;\n" +
            "    vTexCoord = (uTexMatrix * aTexCoord).xy;\n" +
            "}\n";

    private static final String COPY_OES_FRAGMENT_SHADER =
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision mediump float;\n" +
            "varying vec2 vTexCoord;\n" +
            "uniform samplerExternalOES uTexture;\n" +
            "void main() {\n" +
            "    gl_FragColor = texture2D(uTexture, vTexCoord);\n" +
            "}\n";

    private static final String PASS_VERTEX_SHADER =
            "#version 300 es\n" +
            "in vec4 aPosition;\n" +
            "void main() {\n" +
            "    gl_Position = aPosition;\n" +
            "}\n";

    // FSR 1 EASU, FP32 path ported from ffx_fsr1.h (using texelFetch instead of gather)
    private static final String EASU_FRAGMENT_SHADER =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "uniform highp sampler2D uInput;\n" +
            "uniform vec4 uCon0;\n" +
            "uniform ivec2 uInputSize;\n" +
            "out vec4 fragColor;\n" +
            "vec3 fetchC(ivec2 p) {\n" +
            "    return texelFetch(uInput, clamp(p, ivec2(0), uInputSize - ivec2(1)), 0).rgb;\n" +
            "}\n" +
            "float lumaOf(vec3 c) { return c.b * 0.5 + (c.r * 0.5 + c.g); }\n" +
            "void easuTap(inout vec3 aC, inout float aW, vec2 off, vec2 dir, vec2 len, float lob, float clp, vec3 c) {\n" +
            "    vec2 v;\n" +
            "    v.x = (off.x * ( dir.x)) + (off.y * dir.y);\n" +
            "    v.y = (off.x * (-dir.y)) + (off.y * dir.x);\n" +
            "    v *= len;\n" +
            "    float d2 = v.x * v.x + v.y * v.y;\n" +
            "    d2 = min(d2, clp);\n" +
            "    float wB = (2.0 / 5.0) * d2 - 1.0;\n" +
            "    float wA = lob * d2 - 1.0;\n" +
            "    wB *= wB;\n" +
            "    wA *= wA;\n" +
            "    wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);\n" +
            "    float w = wB * wA;\n" +
            "    aC += c * w;\n" +
            "    aW += w;\n" +
            "}\n" +
            "void easuSet(inout vec2 dir, inout float len, float w, float lA, float lB, float lC, float lD, float lE) {\n" +
            "    float dc = lD - lC;\n" +
            "    float cb = lC - lB;\n" +
            "    float lenX = max(abs(dc), abs(cb));\n" +
            "    lenX = 1.0 / max(lenX, 1.0 / 32768.0);\n" +
            "    float dirX = lD - lB;\n" +
            "    dir.x += dirX * w;\n" +
            "    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);\n" +
            "    lenX *= lenX;\n" +
            "    len += lenX * w;\n" +
            "    float ec = lE - lC;\n" +
            "    float ca = lC - lA;\n" +
            "    float lenY = max(abs(ec), abs(ca));\n" +
            "    lenY = 1.0 / max(lenY, 1.0 / 32768.0);\n" +
            "    float dirY = lE - lA;\n" +
            "    dir.y += dirY * w;\n" +
            "    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);\n" +
            "    lenY *= lenY;\n" +
            "    len += lenY * w;\n" +
            "}\n" +
            "void main() {\n" +
            "    vec2 ip = floor(gl_FragCoord.xy);\n" +
            "    vec2 pp = ip * uCon0.xy + uCon0.zw;\n" +
            "    vec2 fp = floor(pp);\n" +
            "    pp -= fp;\n" +
            "    ivec2 f0 = ivec2(fp);\n" +
            "    vec3 bC = fetchC(f0 + ivec2( 0, -1));\n" +
            "    vec3 cC = fetchC(f0 + ivec2( 1, -1));\n" +
            "    vec3 eC = fetchC(f0 + ivec2(-1,  0));\n" +
            "    vec3 fC = fetchC(f0);\n" +
            "    vec3 gC = fetchC(f0 + ivec2( 1,  0));\n" +
            "    vec3 hC = fetchC(f0 + ivec2( 2,  0));\n" +
            "    vec3 iC = fetchC(f0 + ivec2(-1,  1));\n" +
            "    vec3 jC = fetchC(f0 + ivec2( 0,  1));\n" +
            "    vec3 kC = fetchC(f0 + ivec2( 1,  1));\n" +
            "    vec3 lC = fetchC(f0 + ivec2( 2,  1));\n" +
            "    vec3 nC = fetchC(f0 + ivec2( 0,  2));\n" +
            "    vec3 oC = fetchC(f0 + ivec2( 1,  2));\n" +
            "    float bL = lumaOf(bC); float cL = lumaOf(cC); float eL = lumaOf(eC); float fL = lumaOf(fC);\n" +
            "    float gL = lumaOf(gC); float hL = lumaOf(hC); float iL = lumaOf(iC); float jL = lumaOf(jC);\n" +
            "    float kL = lumaOf(kC); float lL = lumaOf(lC); float nL = lumaOf(nC); float oL = lumaOf(oC);\n" +
            "    vec2 dir = vec2(0.0);\n" +
            "    float len = 0.0;\n" +
            "    easuSet(dir, len, (1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL);\n" +
            "    easuSet(dir, len, pp.x * (1.0 - pp.y), cL, fL, gL, hL, kL);\n" +
            "    easuSet(dir, len, (1.0 - pp.x) * pp.y, fL, iL, jL, kL, nL);\n" +
            "    easuSet(dir, len, pp.x * pp.y, gL, jL, kL, lL, oL);\n" +
            "    vec2 dir2 = dir * dir;\n" +
            "    float dirR = dir2.x + dir2.y;\n" +
            "    bool zro = dirR < (1.0 / 32768.0);\n" +
            "    dirR = inversesqrt(max(dirR, 1.0 / 32768.0));\n" +
            "    dirR = zro ? 1.0 : dirR;\n" +
            "    dir.x = zro ? 1.0 : dir.x;\n" +
            "    dir *= dirR;\n" +
            "    len = len * 0.5;\n" +
            "    len *= len;\n" +
            "    float stretch = (dir.x * dir.x + dir.y * dir.y) / max(max(abs(dir.x), abs(dir.y)), 1.0 / 32768.0);\n" +
            "    vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);\n" +
            "    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;\n" +
            "    float clp = 1.0 / lob;\n" +
            "    vec3 min4 = min(min(fC, gC), min(jC, kC));\n" +
            "    vec3 max4 = max(max(fC, gC), max(jC, kC));\n" +
            "    vec3 aC = vec3(0.0);\n" +
            "    float aW = 0.0;\n" +
            "    easuTap(aC, aW, vec2( 0.0, -1.0) - pp, dir, len2, lob, clp, bC);\n" +
            "    easuTap(aC, aW, vec2( 1.0, -1.0) - pp, dir, len2, lob, clp, cC);\n" +
            "    easuTap(aC, aW, vec2(-1.0,  1.0) - pp, dir, len2, lob, clp, iC);\n" +
            "    easuTap(aC, aW, vec2( 0.0,  1.0) - pp, dir, len2, lob, clp, jC);\n" +
            "    easuTap(aC, aW, vec2( 0.0,  0.0) - pp, dir, len2, lob, clp, fC);\n" +
            "    easuTap(aC, aW, vec2(-1.0,  0.0) - pp, dir, len2, lob, clp, eC);\n" +
            "    easuTap(aC, aW, vec2( 1.0,  1.0) - pp, dir, len2, lob, clp, kC);\n" +
            "    easuTap(aC, aW, vec2( 2.0,  1.0) - pp, dir, len2, lob, clp, lC);\n" +
            "    easuTap(aC, aW, vec2( 2.0,  0.0) - pp, dir, len2, lob, clp, hC);\n" +
            "    easuTap(aC, aW, vec2( 1.0,  0.0) - pp, dir, len2, lob, clp, gC);\n" +
            "    easuTap(aC, aW, vec2( 1.0,  2.0) - pp, dir, len2, lob, clp, oC);\n" +
            "    easuTap(aC, aW, vec2( 0.0,  2.0) - pp, dir, len2, lob, clp, nC);\n" +
            "    fragColor = vec4(min(max4, max(min4, aC / aW)), 1.0);\n" +
            "}\n";

    // FSR 1 RCAS with denoise, FP32 path ported from ffx_fsr1.h
    private static final String RCAS_FRAGMENT_SHADER =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "uniform highp sampler2D uInput;\n" +
            "uniform float uSharpness;\n" +
            "uniform ivec2 uInputSize;\n" +
            "out vec4 fragColor;\n" +
            "vec3 fetchC(ivec2 p) {\n" +
            "    return texelFetch(uInput, clamp(p, ivec2(0), uInputSize - ivec2(1)), 0).rgb;\n" +
            "}\n" +
            "void main() {\n" +
            "    ivec2 sp = ivec2(gl_FragCoord.xy);\n" +
            "    vec3 b = fetchC(sp + ivec2( 0, -1));\n" +
            "    vec3 d = fetchC(sp + ivec2(-1,  0));\n" +
            "    vec3 e = fetchC(sp);\n" +
            "    vec3 f = fetchC(sp + ivec2( 1,  0));\n" +
            "    vec3 h = fetchC(sp + ivec2( 0,  1));\n" +
            "    if (uSharpness <= 0.0) { fragColor = vec4(e, 1.0); return; }\n" +
            "    float bL = b.b * 0.5 + (b.r * 0.5 + b.g);\n" +
            "    float dL = d.b * 0.5 + (d.r * 0.5 + d.g);\n" +
            "    float eL = e.b * 0.5 + (e.r * 0.5 + e.g);\n" +
            "    float fL = f.b * 0.5 + (f.r * 0.5 + f.g);\n" +
            "    float hL = h.b * 0.5 + (h.r * 0.5 + h.g);\n" +
            "    float nz = 0.25 * bL + 0.25 * dL + 0.25 * fL + 0.25 * hL - eL;\n" +
            "    float rangeL = max(max(max(bL, dL), max(eL, fL)), hL) - min(min(min(bL, dL), min(eL, fL)), hL);\n" +
            "    nz = clamp(abs(nz) / max(rangeL, 1.0 / 32768.0), 0.0, 1.0);\n" +
            "    nz = -0.5 * nz + 1.0;\n" +
            "    vec3 mn4 = min(min(b, d), min(f, h));\n" +
            "    vec3 mx4 = max(max(b, d), max(f, h));\n" +
            "    vec3 hitMin = min(mn4, e) / max(4.0 * mx4, vec3(1.0 / 32768.0));\n" +
            "    vec3 hitMax = (1.0 - max(mx4, e)) / min(4.0 * mn4 - 4.0, vec3(-1.0 / 32768.0));\n" +
            "    vec3 lobeRGB = max(-hitMin, hitMax);\n" +
            "    float lobe = max(-(0.25 - (1.0 / 16.0)), min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * uSharpness;\n" +
            "    lobe *= nz;\n" +
            "    float rcpL = 1.0 / (4.0 * lobe + 1.0);\n" +
            "    fragColor = vec4((lobe * b + lobe * d + lobe * h + lobe * f + e) * rcpL, 1.0);\n" +
            "}\n";

    private static final float[] QUAD_POSITIONS = {
            -1.0f, -1.0f,
             1.0f, -1.0f,
            -1.0f,  1.0f,
             1.0f,  1.0f,
    };

    private static final float[] QUAD_TEXCOORDS = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f,
    };

    private static final float[] IDENTITY_MATRIX = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
    };

    // Maps (s, t) to (s, 1 - t): MWSR's vertex shader expects the texture's first row to be
    // the top of the picture, while GL renders the bottom row first (column-major matrix)
    private static final float[] FLIP_Y_MATRIX = {
            1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, 1, 0,
            0, 1, 0, 1,
    };

    /** Upscaler choices of the "list_gfn_upscaler" preference. */
    public static final int UPSCALER_MWSR = 0;
    public static final int UPSCALER_FSR = 1;

    private static final String[] MWSR_UNIFORMS = {
            "uTexelScale", "uDimsAcross", "uKernel", "uZone", "uZoneBonus", "uBound"
    };

    private final int streamWidth;
    private final int streamHeight;
    private final float sharpness;
    private final boolean lowLatency;

    // MWSR shader sources (null if MWSR isn't used), read from the assets on the caller's thread
    private final String mwsrVertexSource;
    private final String mwsrUpSource, mwsrDown4Source, mwsrDown9Source, mwsrDown16Source;

    // Shown in the performance overlay; written on the GL thread
    private volatile String statusText = "Bildverbesserung: wird vorbereitet";

    private final HandlerThread glThread;
    private final Handler glHandler;
    private final AtomicInteger pendingFrames = new AtomicInteger();

    // Only accessed on the GL thread
    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLConfig eglConfig;
    private EGLSurface windowSurface = EGL14.EGL_NO_SURFACE;
    private EGLSurface pbufferSurface = EGL14.EGL_NO_SURFACE;
    private SurfaceTexture surfaceTexture;
    private Surface inputSurface;
    private int oesTexture;
    private int copyProgram, easuProgram, rcasProgram, copyRgbProgram;
    private int mwsrUpProgram, mwsrDown4Program, mwsrDown9Program, mwsrDown16Program;
    private final float[] flippedTexMatrix = new float[16];
    private float[] mwsrUniforms;
    private int mwsrUniformsOutW, mwsrUniformsOutH;
    private int sourceTexture, sourceFbo;
    private int upscaledTexture, upscaledFbo;
    private int upscaledWidth, upscaledHeight;
    private int outputWidth, outputHeight;
    private final float[] texMatrix = new float[16];
    private FloatBuffer positionBuffer, texCoordBuffer;
    private boolean released;
    private boolean loggedMode;

    private GlUpscaleRenderer(int streamWidth, int streamHeight, float sharpness, boolean lowLatency,
                              AssetManager assets, boolean useMwsr) {
        this.streamWidth = streamWidth;
        this.streamHeight = streamHeight;
        this.sharpness = Math.max(0.0f, Math.min(sharpness, 1.0f));
        this.lowLatency = lowLatency;

        String vs = null, up = null, d4 = null, d9 = null, d16 = null;
        if (useMwsr && assets != null) {
            try {
                vs = readAsset(assets, "mwsr/mwsr.vert");
                up = readAsset(assets, "mwsr/mwsr-up.frag");
                d4 = readAsset(assets, "mwsr/mwsr-down-4.frag");
                d9 = readAsset(assets, "mwsr/mwsr-down-9.frag");
                d16 = readAsset(assets, "mwsr/mwsr-down-16.frag");
            } catch (IOException e) {
                LimeLog.warning("GFN: Unable to read the MWSR shaders, using FSR: " + e);
                vs = null;
            }
        }
        this.mwsrVertexSource = vs;
        this.mwsrUpSource = up;
        this.mwsrDown4Source = d4;
        this.mwsrDown9Source = d9;
        this.mwsrDown16Source = d16;

        glThread = new HandlerThread("GFN-GL", Process.THREAD_PRIORITY_URGENT_DISPLAY);
        glThread.start();
        glHandler = new Handler(glThread.getLooper());
    }

    /**
     * Creates the renderer. Returns null if OpenGL ES 3.0 post-processing is not available,
     * in which case the caller should render directly to the SurfaceView as usual.
     */
    public static GlUpscaleRenderer create(Surface outputSurface, int outputWidth, int outputHeight,
                                           int streamWidth, int streamHeight,
                                           float sharpness, boolean lowLatency,
                                           AssetManager assets, int upscaler) {
        final GlUpscaleRenderer renderer = new GlUpscaleRenderer(streamWidth, streamHeight, sharpness, lowLatency,
                assets, upscaler == UPSCALER_MWSR);
        final boolean[] success = new boolean[1];

        renderer.runOnGlThread(() -> {
            try {
                success[0] = renderer.initGl(outputSurface, outputWidth, outputHeight);
            } catch (RuntimeException e) {
                LimeLog.warning("GFN: GL post-processing init failed: " + e);
                success[0] = false;
            }
        });

        if (!success[0]) {
            renderer.release();
            return null;
        }

        return renderer;
    }

    /** The surface MediaCodec should decode into. */
    public Surface getInputSurface() {
        return inputSurface;
    }

    /** One line for the performance overlay: whether and how the picture is upscaled. */
    public String getStatusText() {
        return statusText;
    }

    private static String readAsset(AssetManager assets, String name) throws IOException {
        try (InputStream in = assets.open(name)) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int read;
            while ((read = in.read(buffer)) > 0) {
                out.write(buffer, 0, read);
            }
            return new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    /** Called when the SurfaceView changes size. */
    public void setOutputSize(final int width, final int height) {
        runOnGlThread(() -> {
            outputWidth = width;
            outputHeight = height;
            loggedMode = false;
        });
    }

    /**
     * Stops drawing into the SurfaceView. Must be called from surfaceDestroyed()
     * because the output surface is invalid once that callback returns.
     */
    public void releaseOutput() {
        runOnGlThread(() -> {
            if (eglDisplay != EGL14.EGL_NO_DISPLAY && windowSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglMakeCurrent(eglDisplay, pbufferSurface, pbufferSurface, eglContext);
                EGL14.eglDestroySurface(eglDisplay, windowSurface);
                windowSurface = EGL14.EGL_NO_SURFACE;
            }
        });
    }

    /** Releases all resources. */
    public void release() {
        runOnGlThread(this::releaseGl);
        glThread.quitSafely();
    }

    private void runOnGlThread(final Runnable runnable) {
        if (Thread.currentThread() == glThread) {
            runnable.run();
            return;
        }

        final CountDownLatch latch = new CountDownLatch(1);
        boolean posted = glHandler.post(() -> {
            try {
                runnable.run();
            } finally {
                latch.countDown();
            }
        });

        if (!posted) {
            return;
        }

        try {
            if (!latch.await(2, TimeUnit.SECONDS)) {
                LimeLog.warning("GFN: GL thread did not respond in time");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private boolean initGl(Surface outputSurface, int width, int height) {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            return false;
        }

        int[] version = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            eglDisplay = EGL14.EGL_NO_DISPLAY;
            return false;
        }

        int[] configAttribs = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGLExt.EGL_OPENGL_ES3_BIT_KHR,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT | EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0) || numConfigs[0] == 0) {
            LimeLog.warning("GFN: No suitable EGL config for OpenGL ES 3.0");
            return false;
        }
        eglConfig = configs[0];

        int[] contextAttribs = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE
        };
        eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        if (eglContext == null || eglContext == EGL14.EGL_NO_CONTEXT) {
            eglContext = EGL14.EGL_NO_CONTEXT;
            LimeLog.warning("GFN: Unable to create an OpenGL ES 3.0 context");
            return false;
        }

        int[] pbufferAttribs = {
                EGL14.EGL_WIDTH, 1,
                EGL14.EGL_HEIGHT, 1,
                EGL14.EGL_NONE
        };
        pbufferSurface = EGL14.eglCreatePbufferSurface(eglDisplay, eglConfig, pbufferAttribs, 0);
        if (pbufferSurface == null || pbufferSurface == EGL14.EGL_NO_SURFACE) {
            pbufferSurface = EGL14.EGL_NO_SURFACE;
            return false;
        }

        windowSurface = EGL14.eglCreateWindowSurface(eglDisplay, eglConfig, outputSurface, new int[] { EGL14.EGL_NONE }, 0);
        if (windowSurface == null || windowSurface == EGL14.EGL_NO_SURFACE) {
            windowSurface = EGL14.EGL_NO_SURFACE;
            LimeLog.warning("GFN: Unable to create EGL window surface");
            return false;
        }

        if (!EGL14.eglMakeCurrent(eglDisplay, windowSurface, windowSurface, eglContext)) {
            return false;
        }

        if (lowLatency) {
            // Don't block on V-sync when presenting (if the driver supports it)
            EGL14.eglSwapInterval(eglDisplay, 0);
        }

        copyProgram = buildProgram(COPY_VERTEX_SHADER, COPY_OES_FRAGMENT_SHADER);
        easuProgram = buildProgram(PASS_VERTEX_SHADER, EASU_FRAGMENT_SHADER);
        rcasProgram = buildProgram(PASS_VERTEX_SHADER, RCAS_FRAGMENT_SHADER);
        copyRgbProgram = buildProgram(COPY_VERTEX_SHADER,
                "precision mediump float;\n" +
                "varying vec2 vTexCoord;\n" +
                "uniform sampler2D uTexture;\n" +
                "void main() {\n" +
                "    gl_FragColor = texture2D(uTexture, vTexCoord);\n" +
                "}\n");
        if (copyProgram == 0 || easuProgram == 0 || rcasProgram == 0 || copyRgbProgram == 0) {
            return false;
        }

        if (mwsrVertexSource != null) {
            mwsrUpProgram = buildProgram(mwsrVertexSource, mwsrUpSource);
            mwsrDown4Program = buildProgram(mwsrVertexSource, mwsrDown4Source);
            mwsrDown9Program = buildProgram(mwsrVertexSource, mwsrDown9Source);
            mwsrDown16Program = buildProgram(mwsrVertexSource, mwsrDown16Source);
            if (mwsrUpProgram == 0) {
                LimeLog.warning("GFN: MWSR shaders unavailable on this GPU, using FSR");
            }
        }

        positionBuffer = createFloatBuffer(QUAD_POSITIONS);
        texCoordBuffer = createFloatBuffer(QUAD_TEXCOORDS);

        // External texture for MediaCodec output
        int[] tex = new int[1];
        GLES20.glGenTextures(1, tex, 0);
        oesTexture = tex[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTexture);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        // RGB copy of the frame at stream resolution
        int[] fbo = new int[1];
        sourceTexture = createRenderTexture(streamWidth, streamHeight);
        GLES20.glGenFramebuffers(1, fbo, 0);
        sourceFbo = fbo[0];
        if (!attachTexture(sourceFbo, sourceTexture)) {
            return false;
        }

        surfaceTexture = new SurfaceTexture(oesTexture);
        surfaceTexture.setDefaultBufferSize(streamWidth, streamHeight);
        surfaceTexture.setOnFrameAvailableListener(this, glHandler);
        inputSurface = new Surface(surfaceTexture);

        outputWidth = width;
        outputHeight = height;

        LimeLog.info("GFN: GL post-processing initialized (stream " + streamWidth + "x" + streamHeight +
                ", output " + width + "x" + height + ", sharpness " + sharpness + ")");
        return true;
    }

    @Override
    public void onFrameAvailable(SurfaceTexture st) {
        // Coalesce frames so we always draw only the newest one
        if (pendingFrames.getAndIncrement() == 0) {
            glHandler.post(this::drawFrame);
        }
    }

    private void drawFrame() {
        int frames = pendingFrames.getAndSet(0);
        if (released || surfaceTexture == null) {
            return;
        }

        if (windowSurface == EGL14.EGL_NO_SURFACE) {
            // No output: just consume the frames so the decoder doesn't stall
            EGL14.eglMakeCurrent(eglDisplay, pbufferSurface, pbufferSurface, eglContext);
            for (int i = 0; i < frames; i++) {
                surfaceTexture.updateTexImage();
            }
            return;
        }

        EGL14.eglMakeCurrent(eglDisplay, windowSurface, windowSurface, eglContext);

        try {
            for (int i = 0; i < frames; i++) {
                surfaceTexture.updateTexImage();
            }
        } catch (RuntimeException e) {
            LimeLog.warning("GFN: updateTexImage() failed: " + e);
            return;
        }
        surfaceTexture.getTransformMatrix(texMatrix);

        int outW = outputWidth;
        int outH = outputHeight;
        boolean upscale = outW >= streamWidth && outH >= streamHeight &&
                (outW > streamWidth || outH > streamHeight);
        boolean oneToOne = outW == streamWidth && outH == streamHeight;

        // MWSR handles both directions in one pass; at 1:1 only the sharpening applies
        int mwsrProgram = oneToOne ? 0 : getMwsrProgram(outW, outH);

        if (!loggedMode) {
            String mode;
            if (mwsrProgram != 0) {
                mode = upscale ? "MWSR" : "MWSR reduction (" + MwsrLaw.taps(streamWidth, streamHeight, outW, outH) + " samples)";
            }
            else {
                mode = upscale ? "FSR EASU + RCAS" : (oneToOne ? "RCAS" : "bilinear");
            }
            LimeLog.info("GFN: GL post-processing mode: " + mode +
                    " (" + streamWidth + "x" + streamHeight + " -> " + outW + "x" + outH + ")");
            updateStatus(mwsrProgram != 0, upscale, oneToOne, outW, outH);
            loggedMode = true;
        }

        if (mwsrProgram != 0) {
            // Pass 1: external OES texture -> RGB texture at stream resolution, first row = top
            Matrix.multiplyMM(flippedTexMatrix, 0, texMatrix, 0, FLIP_Y_MATRIX, 0);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, sourceFbo);
            GLES20.glViewport(0, 0, streamWidth, streamHeight);
            drawTexturedQuad(copyProgram, GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTexture, flippedTexMatrix);

            // Pass 2: MWSR straight into the window surface (full-screen triangle)
            if (mwsrUniforms == null || mwsrUniformsOutW != outW || mwsrUniformsOutH != outH) {
                mwsrUniforms = MwsrLaw.uniforms(streamWidth, streamHeight, outW, outH);
                mwsrUniformsOutW = outW;
                mwsrUniformsOutH = outH;
            }

            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
            GLES20.glViewport(0, 0, outW, outH);
            GLES20.glUseProgram(mwsrProgram);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sourceTexture);
            GLES20.glUniform1i(GLES20.glGetUniformLocation(mwsrProgram, "uTex"), 0);
            for (int i = 0; i < MWSR_UNIFORMS.length; i++) {
                int loc = GLES20.glGetUniformLocation(mwsrProgram, MWSR_UNIFORMS[i]);
                if (loc >= 0) {
                    GLES20.glUniform4fv(loc, 1, mwsrUniforms, i * 4);
                }
            }
            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 3);
        }
        else {
            // Pass 1: external OES texture -> RGB texture at stream resolution
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, sourceFbo);
            GLES20.glViewport(0, 0, streamWidth, streamHeight);
            drawTexturedQuad(copyProgram, GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTexture, texMatrix);

            int rcasInput = sourceTexture;

            if (upscale) {
                // Pass 2: FSR EASU to output resolution
                if (!ensureUpscaledTarget(outW, outH)) {
                    return;
                }

                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, upscaledFbo);
                GLES20.glViewport(0, 0, outW, outH);
                GLES20.glUseProgram(easuProgram);
                GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, sourceTexture);
                GLES20.glUniform1i(GLES20.glGetUniformLocation(easuProgram, "uInput"), 0);
                float scaleX = (float) streamWidth / outW;
                float scaleY = (float) streamHeight / outH;
                GLES20.glUniform4f(GLES20.glGetUniformLocation(easuProgram, "uCon0"),
                        scaleX, scaleY, 0.5f * scaleX - 0.5f, 0.5f * scaleY - 0.5f);
                GLES20.glUniform2i(GLES20.glGetUniformLocation(easuProgram, "uInputSize"), streamWidth, streamHeight);
                drawQuad(easuProgram);

                rcasInput = upscaledTexture;
            }

            // Pass 3: into the window surface
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
            GLES20.glViewport(0, 0, outW, outH);

            if (upscale || oneToOne) {
                GLES20.glUseProgram(rcasProgram);
                GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, rcasInput);
                GLES20.glUniform1i(GLES20.glGetUniformLocation(rcasProgram, "uInput"), 0);
                GLES20.glUniform1f(GLES20.glGetUniformLocation(rcasProgram, "uSharpness"), sharpness);
                GLES20.glUniform2i(GLES20.glGetUniformLocation(rcasProgram, "uInputSize"),
                        upscale ? outW : streamWidth, upscale ? outH : streamHeight);
                drawQuad(rcasProgram);
            }
            else {
                // Downscaling: plain bilinear
                drawTexturedQuad(copyRgbProgram, GLES20.GL_TEXTURE_2D, sourceTexture, IDENTITY_MATRIX);
            }
        }

        if (!EGL14.eglSwapBuffers(eglDisplay, windowSurface)) {
            LimeLog.warning("GFN: eglSwapBuffers() failed: 0x" + Integer.toHexString(EGL14.eglGetError()));
        }
    }

    // The MWSR program for the current output size, or 0 if MWSR isn't used/available
    private int getMwsrProgram(int outW, int outH) {
        if (mwsrUpProgram == 0) {
            return 0;
        }

        switch (MwsrLaw.mode(streamWidth, streamHeight, outW, outH)) {
            case MwsrLaw.MODE_UP:
                return mwsrUpProgram;
            case MwsrLaw.MODE_DOWN:
                switch (MwsrLaw.taps(streamWidth, streamHeight, outW, outH)) {
                    case 4:
                        return mwsrDown4Program;
                    case 9:
                        return mwsrDown9Program;
                    default:
                        return mwsrDown16Program;
                }
            default:
                return 0;
        }
    }

    private void updateStatus(boolean mwsr, boolean upscale, boolean oneToOne, int outW, int outH) {
        String sizes = streamWidth + "x" + streamHeight + " \u2192 " + outW + "x" + outH;
        int sharpnessPercent = Math.round(sharpness * 100);
        String text;
        if (mwsr) {
            text = upscale ? "Bildverbesserung: MWSR " + sizes
                           : "Bildverbesserung: MWSR verkleinert " + sizes;
        }
        else if (upscale) {
            text = "Bildverbesserung: AMD FSR 1 " + sizes + ", Schärfe " + sharpnessPercent + " %";
        }
        else if (oneToOne) {
            text = sharpnessPercent > 0
                    ? "Bildverbesserung: nur Schärfe " + sharpnessPercent + " % (Stream = Bildschirm)"
                    : "Bildverbesserung: aus (Stream = Bildschirm, Schärfe 0 %)";
        }
        else {
            text = "Bildverbesserung: bilinear verkleinert " + sizes;
        }
        statusText = text;
    }

    private boolean ensureUpscaledTarget(int width, int height) {
        if (upscaledTexture != 0 && upscaledWidth == width && upscaledHeight == height) {
            return true;
        }

        if (upscaledTexture != 0) {
            GLES20.glDeleteTextures(1, new int[] { upscaledTexture }, 0);
            upscaledTexture = 0;
        }
        if (upscaledFbo == 0) {
            int[] fbo = new int[1];
            GLES20.glGenFramebuffers(1, fbo, 0);
            upscaledFbo = fbo[0];
        }

        upscaledTexture = createRenderTexture(width, height);
        upscaledWidth = width;
        upscaledHeight = height;
        return attachTexture(upscaledFbo, upscaledTexture);
    }

    private void drawTexturedQuad(int program, int target, int texture, float[] matrix) {
        GLES20.glUseProgram(program);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(target, texture);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uTexture"), 0);
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(program, "uTexMatrix"), 1, false, matrix, 0);

        int texCoordLoc = GLES20.glGetAttribLocation(program, "aTexCoord");
        GLES20.glEnableVertexAttribArray(texCoordLoc);
        GLES20.glVertexAttribPointer(texCoordLoc, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer);

        drawQuad(program);

        GLES20.glDisableVertexAttribArray(texCoordLoc);
    }

    private void drawQuad(int program) {
        int positionLoc = GLES20.glGetAttribLocation(program, "aPosition");
        GLES20.glEnableVertexAttribArray(positionLoc);
        GLES20.glVertexAttribPointer(positionLoc, 2, GLES20.GL_FLOAT, false, 0, positionBuffer);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(positionLoc);
    }

    private static int createRenderTexture(int width, int height) {
        int[] tex = new int[1];
        GLES20.glGenTextures(1, tex, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex[0]);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        return tex[0];
    }

    private static boolean attachTexture(int fbo, int texture) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, texture, 0);
        int status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            LimeLog.warning("GFN: Framebuffer incomplete: 0x" + Integer.toHexString(status));
            return false;
        }
        return true;
    }

    private static FloatBuffer createFloatBuffer(float[] data) {
        FloatBuffer buffer = ByteBuffer.allocateDirect(data.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        buffer.put(data).position(0);
        return buffer;
    }

    private static int compileShader(int type, String source) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);
        int[] status = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0);
        if (status[0] == 0) {
            LimeLog.warning("GFN: Shader compile failed: " + GLES20.glGetShaderInfoLog(shader));
            GLES20.glDeleteShader(shader);
            return 0;
        }
        return shader;
    }

    private static int buildProgram(String vertexSource, String fragmentSource) {
        int vs = compileShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        int fs = compileShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        if (vs == 0 || fs == 0) {
            return 0;
        }

        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vs);
        GLES20.glAttachShader(program, fs);
        GLES20.glLinkProgram(program);
        GLES20.glDeleteShader(vs);
        GLES20.glDeleteShader(fs);

        int[] status = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, status, 0);
        if (status[0] == 0) {
            LimeLog.warning("GFN: Program link failed: " + GLES20.glGetProgramInfoLog(program));
            GLES20.glDeleteProgram(program);
            return 0;
        }
        return program;
    }

    private void releaseGl() {
        released = true;

        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            if (pbufferSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglMakeCurrent(eglDisplay, pbufferSurface, pbufferSurface, eglContext);
            }

            if (surfaceTexture != null) {
                surfaceTexture.setOnFrameAvailableListener(null);
                surfaceTexture.release();
                surfaceTexture = null;
            }
            if (inputSurface != null) {
                inputSurface.release();
                inputSurface = null;
            }

            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
            if (windowSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, windowSurface);
                windowSurface = EGL14.EGL_NO_SURFACE;
            }
            if (pbufferSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, pbufferSurface);
                pbufferSurface = EGL14.EGL_NO_SURFACE;
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                // Destroying the context also frees all GL objects created in it
                EGL14.eglDestroyContext(eglDisplay, eglContext);
                eglContext = EGL14.EGL_NO_CONTEXT;
            }
            // NB: We don't call eglTerminate() because the default display is shared
            // with the UI toolkit's renderer in this process.
            EGL14.eglReleaseThread();
            eglDisplay = EGL14.EGL_NO_DISPLAY;
        }
    }
}
