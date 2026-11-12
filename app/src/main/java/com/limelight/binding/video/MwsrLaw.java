package com.limelight.binding.video;

/**
 * GFN fork: the CPU side of MWSR (MoonlightWeb Super Resolution) for the GLSL ES shaders in
 * assets/mwsr. A line-for-line Java port of mwsr/cpp/mwsr.h (regime(), law(), downLaw() and
 * constants()) from https://github.com/linckosz/moonlight-web, without the gaze zone.
 *
 * MWSR is Copyright (c) 2026 Bruno Martin and licensed under the MIT License
 * (see assets/mwsr/LICENSE-MWSR).
 */
public final class MwsrLaw {
    public static final int MODE_COPY = 0;
    public static final int MODE_UP = 1;
    public static final int MODE_DOWN = 2;

    // value = c0 + c1 x + c2 x^2 with x = ln(scale), for kc, kg, kr and kl
    private static final double[][] LAW = {
            {0.6353583202388096, 0.021845681633595574, 0.12781268100437676},  // kc
            {-0.1288413863403059, 2.5844683855070816, -1.260092021482419},    // kg
            {1.0071546009267338, -1.0209066704914191, 0.5699605831985769},    // kr
            {0.33416994543071277, -0.2962022126856069, 0.08628272130287401},  // kl
    };
    private static final double LAW_SCALE_MIN = 1.07, LAW_SCALE_MAX = 4.0;
    private static final double FORCE_FROM = 1.3333333333333333, FORCE_TO = 2.0, FORCE_TOP = 1.35;
    private static final double LAW_FT = 0.005, LAW_KWC = 0.1;
    private static final double FULL_SCALE = 1.07;

    // The reduction's grids for the shader's own light (LINEAR_LIGHT 1): {minimum scale, samples}
    private static final double[][] DOWN_GRIDS = {
            {0.5, 4},
            {0.25, 9},
            {0.0, 16},
    };

    private static final double[] DOWN_SCALE = {0.5, 0.6666666666666666};
    private static final double[] DOWN_GAIN = {0.25, 0.45};
    private static final double[] DOWN_EDGE_GAIN = {0.05, 0.2};
    private static final double[] DOWN_MARGIN = {0.1, 0.25};
    private static final double[] DOWN_EDGE_MARGIN = {0.0, 0.1};
    private static final double DOWN_FADE = 0.93, DOWN_FT = 0.003;

    private MwsrLaw() {}

    /** The mode for a frame of inW x inH drawn at outW x outH (one of the MODE_* values). */
    public static int mode(int inW, int inH, int outW, int outH) {
        if (inW == outW && inH == outH) {
            return MODE_COPY;
        }
        double s = Math.min((double) outW / inW, (double) outH / inH);
        return s >= 1.0 ? MODE_UP : MODE_DOWN;
    }

    /** The reduction grid's samples (4, 9 or 16) for a downscale, 0 otherwise. */
    public static int taps(int inW, int inH, int outW, int outH) {
        if (mode(inW, inH, outW, outH) != MODE_DOWN) {
            return 0;
        }
        double s = Math.min((double) outW / inW, (double) outH / inH);
        for (double[] grid : DOWN_GRIDS) {
            if (s >= grid[0]) {
                return (int) grid[1];
            }
        }
        return 16;
    }

    private static double hold(double v, double lo, double hi) {
        return Math.min(Math.max(v, lo), hi);
    }

    private static double lawValue(int i, double x) {
        return LAW[i][0] + x * (LAW[i][1] + x * LAW[i][2]);
    }

    private static double lerp(double[] p, double t) {
        return p[0] + (p[1] - p[0]) * t;
    }

    /**
     * The 24 floats of the shaders' uniform block (mwsrUniforms() / mwsr::constants()) for an
     * SDR input of inW x inH drawn at outW x outH, in the order uTexelScale, uDimsAcross,
     * uKernel, uZone, uZoneBonus, uBound (four floats each).
     */
    public static float[] uniforms(int inW, int inH, int outW, int outH) {
        // law()
        double sLaw = hold(Math.min((double) outW / inW, (double) outH / inH), LAW_SCALE_MIN, LAW_SCALE_MAX);
        double x = Math.log(sLaw);
        double force = 1.0 + (FORCE_TOP - 1.0) *
                hold(Math.log(sLaw / FORCE_FROM) / Math.log(FORCE_TO / FORCE_FROM), 0.0, 1.0);
        double lawKc = lawValue(0, x);
        double lawKg = lawValue(1, x) * force;
        double lawKr = lawValue(2, x);
        double lawKl = lawValue(3, x) * force;

        // constants()
        double smin = Math.min((double) outW / inW, (double) outH / inH);
        double strength = hold((smin - 1.0) / (FULL_SCALE - 1.0), 0.0, 1.0);
        boolean down = smin < 1.0;

        // downLaw()
        double dFt = DOWN_FT, dKg = 0.0, dKwc = 0.0, dKl = 0.0, dZref = 0.0;
        if (down && smin >= DOWN_SCALE[0]) {
            double t = Math.min((smin - DOWN_SCALE[0]) / (DOWN_SCALE[1] - DOWN_SCALE[0]), 1.0);
            double f = hold((1.0 - smin) / (1.0 - DOWN_FADE), 0.0, 1.0);
            dKg = lerp(DOWN_GAIN, t) * f;
            dKwc = lerp(DOWN_EDGE_GAIN, t) * f;
            dKl = lerp(DOWN_MARGIN, t);
            dZref = lerp(DOWN_EDGE_MARGIN, t);
        }

        double kg = down ? dKg : lawKg * strength;
        double ring = 0.75;
        double zoneFt = 0.005, zoneBoost = 1.5;

        float[] u = new float[24];
        u[0] = (float) (1.0 / inW);
        u[1] = (float) (1.0 / inH);
        u[2] = (float) ((double) outW / inW);
        u[3] = (float) ((double) outH / inH);
        u[4] = (float) inW;
        u[5] = (float) inH;
        u[6] = (float) (lawKr / inW);
        u[7] = (float) (lawKr / inH);
        u[8] = (float) lawKc;
        u[9] = (float) Math.max(down ? dFt : LAW_FT, 1e-6);
        u[10] = (float) kg;
        u[11] = (float) (down ? dKwc : LAW_KWC * strength);
        // The gaze zone (unused: the shaders are built with GAZE 0)
        u[12] = 0.5f;
        u[13] = 0.5f;
        u[14] = 0.2f;
        u[15] = 0.45f;
        u[16] = (float) ring;
        u[17] = (float) zoneFt;
        u[18] = (float) (kg * zoneBoost);
        u[19] = 0.0f;
        u[20] = (float) (down ? dKl : lawKl);
        u[21] = down ? (float) dZref : 1.0f;
        u[22] = 0.0f;
        u[23] = 0.0f;
        return u;
    }
}
