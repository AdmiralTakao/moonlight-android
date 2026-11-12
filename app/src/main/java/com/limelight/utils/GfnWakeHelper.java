package com.limelight.utils;

import android.content.Context;
import android.content.SharedPreferences;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * GFN fork: "Start mit einem Tipp". A wake address is an http(s) URL, optionally prefixed with
 * "POST ", that powers on or wakes a PC: a smart plug (Shelly, Tasmota), UpSnap on a NAS or a
 * Home Assistant webhook. It is stored per PC (by UUID) in its own preferences file.
 */
public final class GfnWakeHelper {
    private static final String PREF_FILENAME = "gfn_wake_urls";

    /** How long the PC list waits for a woken PC to come online. */
    public static final int TIMEOUT_SECONDS = 180;

    /** Wake-on-LAN is repeated at this interval while waiting (the wake address is called once). */
    public static final int WOL_REPEAT_SECONDS = 15;

    private GfnWakeHelper() {}

    public static String getWakeUrl(Context context, String uuid) {
        if (uuid == null) {
            return "";
        }
        return context.getSharedPreferences(PREF_FILENAME, Context.MODE_PRIVATE).getString(uuid, "");
    }

    public static boolean hasWakeUrl(Context context, String uuid) {
        return !getWakeUrl(context, uuid).trim().isEmpty();
    }

    public static void setWakeUrl(Context context, String uuid, String url) {
        if (uuid == null) {
            return;
        }

        SharedPreferences.Editor editor = context.getSharedPreferences(PREF_FILENAME, Context.MODE_PRIVATE).edit();
        if (url == null || url.trim().isEmpty()) {
            editor.remove(uuid);
        }
        else {
            editor.putString(uuid, url.trim());
        }
        editor.apply();
    }

    /**
     * Calls the wake address. Must not be called on the UI thread.
     * Returns a short message for the user; never contains the URL (it may hold credentials).
     */
    public static String callWakeUrl(String wakeUrl) {
        String url = wakeUrl.trim();
        String method = "GET";
        if (url.regionMatches(true, 0, "POST ", 0, 5)) {
            method = "POST";
            url = url.substring(5).trim();
        }
        else if (url.regionMatches(true, 0, "GET ", 0, 4)) {
            url = url.substring(4).trim();
        }

        if (!url.regionMatches(true, 0, "http://", 0, 7) && !url.regionMatches(true, 0, "https://", 0, 8)) {
            return "Ungültige Weck-Adresse (muss mit http:// oder https:// beginnen).";
        }

        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL(url).openConnection();
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(10000);
            conn.setInstanceFollowRedirects(true);
            conn.setUseCaches(false);
            conn.setRequestMethod(method);

            // user:password@host in the URL means HTTP Basic authentication (Shelly, Tasmota)
            String userInfo = conn.getURL().getUserInfo();
            if (userInfo != null && !userInfo.isEmpty()) {
                String auth = android.util.Base64.encodeToString(
                        java.net.URLDecoder.decode(userInfo, "UTF-8").getBytes("UTF-8"),
                        android.util.Base64.NO_WRAP);
                conn.setRequestProperty("Authorization", "Basic " + auth);
            }

            if (method.equals("POST")) {
                conn.setDoOutput(true);
                conn.setFixedLengthStreamingMode(0);
                conn.setRequestProperty("Content-Type", "text/plain");
                conn.getOutputStream().close();
            }

            int status = conn.getResponseCode();

            // Drain the response so the connection can be closed cleanly
            try (InputStream in = status < 400 ? conn.getInputStream() : conn.getErrorStream()) {
                if (in != null) {
                    byte[] buffer = new byte[1024];
                    while (in.read(buffer) > 0) {
                        // Discard
                    }
                }
            }

            if (status >= 200 && status < 400) {
                return "Weck-Adresse aufgerufen (HTTP " + status + ").";
            }
            else {
                return "Weck-Adresse antwortet mit Fehler (HTTP " + status + ").";
            }
        } catch (IOException | IllegalArgumentException e) {
            return "Weck-Adresse nicht erreichbar (" + e.getClass().getSimpleName() + ").";
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }
    }
}
