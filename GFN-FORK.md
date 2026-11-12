# Moonlight GFN-Fork (Android)

Inoffizieller Fork von [Moonlight Android](https://github.com/moonlight-stream/moonlight-android) mit Funktionen
nach dem Vorbild von GeForce NOW. Die Optionen stehen in den Einstellungen unter **„GFN-Features (Fork)“**.

## Bildverbesserung
- **„Bildverbesserung: Upscaling + Schärfe“** einschalten (standardmäßig aus, nicht zusammen mit HDR).
- **Upscaler:** *MWSR* (empfohlen) oder *AMD FSR 1*.
  - MWSR – *MoonlightWeb Super Resolution* von linckosz (MIT): ein Durchgang, auf Handys mehrfach schneller als FSR,
    verkleinert auch sauber. Shader: `app/src/main/assets/mwsr/`, Konstanten: `MwsrLaw.java`.
  - FSR 1: EASU-Upscaling + RCAS-Schärfe (der Schärferegler gilt für FSR und gleiche Auflösung).
- Das **Statistik-Overlay** zeigt in der letzten Zeile, ob und wie gerade hochgerechnet wird.
- Gegen Blockartefakte („pixelig“) hilft kein Upscaler – nur mehr Bitrate und HEVC/AV1 statt H.264.

## Start mit einem Tipp
PC lange antippen → **„Weck-Adresse festlegen…“**. Danach reicht ein Tipp auf den ausgeschalteten PC: Moonlight ruft die
Adresse auf (z. B. Smart-Steckdose, UpSnap, Home-Assistant-Webhook mit vorangestelltem `POST `), sendet zusätzlich
Wake-on-LAN, wartet bis zu 3 Minuten und öffnet dann die App-Liste. Ohne Weck-Adresse gibt es im Menü
**„Wecken und verbinden“** (nur Wake-on-LAN).

Beispiele: `http://192.168.178.50/relay/0?turn=on` (Shelly), `http://192.168.178.51/cm?cmnd=Power%20On` (Tasmota).
Unterwegs muss die Adresse erreichbar sein, z. B. über einen Tailscale-Subnet-Router.

## APK herunterladen
GitHub → **Actions** → neuester erfolgreicher Lauf von „Build APK (GFN fork)“ → *Artifacts* → **Moonlight-GFN-APK-…**.
