/*
  Smart Agricultural Monitoring System (ESP32-Based)
  ----------------------------------------------------
  Sensors : Rain sensor, Soil moisture sensor (YL-69 + LM393),
            DHT11, BMP180, BMP280
  Display : 1.8" ST7735 TFT (128x160, SPI) + Serial Monitor
  Network : Sends telemetry to a Flask API (Render) and shows the
            returned 48h Flood Risk / Rainfall / Temperature-tendency
            predictions in the area that used to show "LET Innovations".

  Everything else (wiring, TFT layout, gauges) is unchanged from the
  original design.

  Libraries required (install via Library Manager):
    - Adafruit GFX Library
    - Adafruit ST7735 and ST7789 Library
    - DHT sensor library (Adafruit)
    - Adafruit Unified Sensor
    - Adafruit BMP085 Library   (covers BMP180)
    - Adafruit BMP280 Library
    - ArduinoJson (by Benoit Blanchon) — v6.x  (v7 also works, syntax below is v6-compatible)
    NOTE: WiFi.h, HTTPClient.h, WiFiClientSecure.h ship with the ESP32 board package,
          nothing extra to install for those.

  Wiring (unchanged from the original design):
    DHT11        DATA -> GPIO4              VCC -> 3.3V   GND -> GND
    Rain Sensor  AO   -> GPIO35 (ADC)       VCC -> 3.3V   GND -> GND   (DO unused)
    Soil Moist.  AO   -> GPIO34 (ADC)       VCC -> 3.3V   GND -> GND   (DO unused)
    BMP180       SDA  -> GPIO21   SCL -> GPIO22   3.3V / GND
    BMP280       SDA  -> GPIO21   SCL -> GPIO22   3.3V / GND
                 (BMP180 and BMP280 must use different I2C addresses)
    ST7735 TFT   CS -> GPIO5   DC -> GPIO15   RST -> GPIO2
                 SCK -> GPIO18  MOSI -> GPIO23
                 VCC -> 3.3V    GND -> GND     BL -> 3.3V

  Server contract:
    POST https://weather-station-prediction.onrender.com/api/telemetry
    Body (JSON): temperature, humidity, temp_bmp180, pressure_loc1,
                 temp_bmp280, pressure_loc2, rain_raw, soil_moisture
                 (rain_raw / soil_moisture are the RAW analogRead() ints, 0-4095)
    Response  : { "predictions_48h": { "flood_risk_percent": 0-100,
                                        "rain_percent": 0-100,
                                        "temperature_percent": 0-100,
                                        "flood_risk_target": "low|watch|high",
                                        "rain_category_target": "dry|light|moderate|heavy",
                                        "temperature_target": 0.0, ... } }
    Every successful POST is what makes the backend append a row to its
    telemetry_history.csv — so "logging to the server" happens automatically
    as a side effect of asking for a prediction. No separate logging call needed.
*/

#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <DHT.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BMP085.h>   // BMP180
#include <Adafruit_BMP280.h>   // BMP280

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <math.h>

// ---------------- WiFi / Server Config ----------------
// Credentials live in arduino_secrets.h, which is git-ignored. Copy
// arduino_secrets.h.example to arduino_secrets.h and fill in your network.
#include "arduino_secrets.h"

const char* SERVER_URL    = "https://weather-station-prediction.onrender.com/api/telemetry";

// How often we push a reading to the server + ask for a fresh prediction.
const unsigned long SEND_INTERVAL = 30000UL; // 30s

// ---------------- Pin Definitions ----------------
#define RAIN_PIN     35   // ADC input, rain sensor AO
#define SOIL_PIN     34   // ADC input, soil moisture AO
#define DHT_PIN      4    // DHT11 data pin

#define TFT_CS       5
#define TFT_DC       15
#define TFT_RST      2
#define TFT_SCK      18
#define TFT_MOSI     23

// BMP280 I2C address (must differ from BMP180's fixed 0x77)
#define BMP280_ADDR  0x76

// ---------------- Heap health / self-healing config ----------------
// getMaxAllocHeap() is the size of the LARGEST single contiguous free block; a
// TLS handshake needs a chunk that size in one piece. Kept as a diagnostic, but
// note it was NOT the cause of the permanent "code: -1" failures on this rig:
// maxAllocBlock stayed flat at ~110KB (2.7x the threshold below) through every
// failure, so a heap watchdog alone could never recover the device.
#define ENABLE_SELF_HEAL_RESTART true
#define HEAP_FRAGMENTATION_THRESHOLD  40960   // 40KB contiguous, typical BearSSL need
#define HEAP_UNHEALTHY_LIMIT          5       // consecutive unhealthy syncs before restart
int consecutiveLowHeapCount = 0;

// Recovery driven by the symptom that actually occurs: syncs failing while the
// heap looks perfectly healthy. Reconnect WiFi after a few failures, reboot if
// even that does not bring it back.
#define FAILED_SYNCS_BEFORE_WIFI_RESET  3
#define FAILED_SYNCS_BEFORE_RESTART     10
int consecutiveFailedSyncs = 0;

// ---------------- Object Instances ----------------
DHT dht(DHT_PIN, DHT11);
Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_RST);
Adafruit_BMP085 bmp180;
Adafruit_BMP280 bmp280;

// One TLS client object, but explicitly reset before every request (see
// sendTelemetryAndGetPrediction). Keeping the object avoids per-request heap
// churn; resetting its socket avoids reusing a connection the server has
// already closed.
WiFiClientSecure secureClient;

bool bmp180_ok = false;
bool bmp280_ok = false;

// ---------------- Sensor Values ----------------
int   rainAnalog   = 0;
int   soilAnalog   = 0;
float dhtHum        = 0.0;
float dhtTemp        = 0.0;
float bmp180Pres     = 0.0;
float bmp180Temp     = 0.0;
float bmp280Pres     = 0.0;
float bmp280Temp     = 0.0;

// ---------------- Prediction Values (from server) ----------------
// Fixed-size buffers rather than Arduino String: a String reassigned every 30s
// for the life of the program reallocates each time and fragments the heap.
char   floodRisk48h[16]    = "";
char   rainCategory48h[16] = "";
float  tempPred48h         = 0.0;
// 0-100 blended scores from the backend (50% seasonal climatology + 50% sensor
// model). Negative means "not supplied", so an older backend still renders via
// the category strings.
float  floodPercent48h     = -1.0;
float  rainPercent48h      = -1.0;
float  tempPercent48h      = -1.0;
bool   havePrediction      = false;
bool   lastSendOk          = false;
unsigned long lastPredictionMillis = 0;

enum TempTendency { TEND_UNKNOWN, TEND_UP, TEND_DOWN, TEND_STABLE };

// ---------------- Timing (non-blocking) ----------------
unsigned long lastReadTime    = 0;
unsigned long lastSendTime    = 0;
const unsigned long READ_INTERVAL = 2000; // ms between local sensor reads/refresh

// ---------------- TFT Layout ----------------
const int HEADER_Y   = 0;
const int SEP_Y       = 9;

const int ROW_Y[4]   = {12, 24, 36, 48};
const int COL1_X     = 2;
const int COL2_X     = 80;
const int VAL_W      = 36;
const int VAL_H       = 10;

const int GAUGE_TITLE_Y = 60;
const int GAUGE_CX[3]   = {27, 80, 133};
const int GAUGE_CY      = 104;
const int GAUGE_R        = 18;
const int GAUGE_VAL_Y    = 108;
const int GAUGE_VAL_W    = 50;

const int STATUS_Y     = 120;
const int STATUS_W     = 156;

bool labelsDrawn = false;

// ---------------- Function Prototypes ----------------
void drawStaticLabels();
void drawHeader();
void drawGaugeTitles();
void readSensors();
void updateDisplayValues();
void updateGauges();
void drawGauge(int idx, float value01, const char* valueText, uint16_t valueColor);
void drawGaugeTrack(int cx, int cy, int r);
void drawNeedle(int cx, int cy, int r, float value01, uint16_t color);
float categoryToValue01(const char* s);
void updateStatusLine(const char* text, uint16_t color);
void printSerialData();
void connectWiFi();
bool sendTelemetryAndGetPrediction();
TempTendency computeTempTendency();
void drawValueBoxW(int x, int y, int w, int h, const char* text, uint16_t color);
void drawValueBox(int x, int y, const char* text, uint16_t color);
void logHeapStatus(const char* label);

void setup() {
  Serial.begin(115200);
  delay(200);

  // Sensors
  dht.begin();

  pinMode(RAIN_PIN, INPUT);
  pinMode(SOIL_PIN, INPUT);

  // I2C bus shared by BMP180 and BMP280
  Wire.begin(21, 22);

  bmp180_ok = bmp180.begin();
  if (!bmp180_ok) {
    Serial.println(F("BMP180 not detected, check wiring/address."));
  }

  bmp280_ok = bmp280.begin(BMP280_ADDR);
  if (!bmp280_ok) {
    Serial.println(F("BMP280 not detected, check wiring/address."));
  }

  // TFT init
  tft.initR(INITR_BLACKTAB);   // adjust tab type if display looks mirrored/offset
  tft.setRotation(1);          // landscape, adjust as needed
  tft.fillScreen(ST77XX_BLACK);

  drawHeader();
  drawStaticLabels();
  drawGaugeTitles();
  labelsDrawn = true;

  secureClient.setInsecure();
  // NOTE: skips TLS certificate verification. Render's endpoint is served over
  // a normal public HTTPS cert, but the ESP32 has no CA bundle loaded by default,
  // so setInsecure() is the common approach for hobby/prototype projects.
  // For production you'd load Render's root CA with secureClient.setCACert(...) instead.

  updateStatusLine("WiFi: connecting...", ST77XX_YELLOW);
  connectWiFi();

  readSensors();
  updateDisplayValues();
  printSerialData();
  lastReadTime = millis();

  logHeapStatus("boot");

  // First telemetry push right away so the screen isn't blank on boot
  updateStatusLine("Syncing...", ST77XX_YELLOW);
  bool ok = sendTelemetryAndGetPrediction();
  updateGauges();
  updateStatusLine(ok ? "Synced" : "Send failed", ok ? ST77XX_GREEN : ST77XX_RED);
  lastSendTime = millis();
}

void loop() {
  unsigned long now = millis();

  // Non-blocking periodic refresh of local sensor readout
  if (now - lastReadTime >= READ_INTERVAL) {
    lastReadTime = now;
    readSensors();
    updateDisplayValues();
    printSerialData();
  }

  // Periodic push to server + prediction refresh
  if (now - lastSendTime >= SEND_INTERVAL) {
    lastSendTime = now;

    if (WiFi.status() != WL_CONNECTED) {
      updateStatusLine("WiFi: reconnecting...", ST77XX_YELLOW);
      connectWiFi();
    }

    updateStatusLine("Syncing...", ST77XX_YELLOW);
    bool ok = sendTelemetryAndGetPrediction();
    updateGauges();

    if (ok) {
      updateStatusLine("Synced", ST77XX_GREEN);
    } else {
      updateStatusLine("Send failed", ST77XX_RED);
    }
  }
}

// ---------------- WiFi ----------------
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to WiFi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000UL) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected, IP: ");
    Serial.println(WiFi.localIP());
    updateStatusLine("WiFi OK", ST77XX_GREEN);
  } else {
    Serial.println("WiFi connection failed (will retry later).");
    updateStatusLine("WiFi FAILED", ST77XX_RED);
  }
}

// ---------------- Sensor Reading ----------------
void readSensors() {
  rainAnalog = analogRead(RAIN_PIN);
  soilAnalog = analogRead(SOIL_PIN);

  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (!isnan(h)) dhtHum = h;
  if (!isnan(t)) dhtTemp = t;

  if (bmp180_ok) {
    bmp180Pres = bmp180.readPressure();       // Pa
    bmp180Temp = bmp180.readTemperature();    // C
  }

  if (bmp280_ok) {
    bmp280Pres = bmp280.readPressure();       // Pa
    bmp280Temp = bmp280.readTemperature();    // C
  }
}

// ---------------- Heap diagnostics ----------------
// getFreeHeap()     = total free heap, scattered across however many blocks.
// getMaxAllocHeap() = size of the LARGEST single contiguous block — this is
//                     what a TLS handshake actually needs in one piece, and
//                     is what will predict connection failures BEFORE
//                     getFreeHeap() looks unhealthy.
// getMinFreeHeap()  = lowest free-heap watermark ever seen since boot.
void logHeapStatus(const char* label) {
  Serial.print("[heap] ");
  Serial.print(label);
  Serial.print(" -> free=");
  Serial.print(ESP.getFreeHeap());
  Serial.print(" maxAllocBlock=");
  Serial.print(ESP.getMaxAllocHeap());
  Serial.print(" minFreeEver=");
  Serial.println(ESP.getMinFreeHeap());
}

// ---------------- Server Communication ----------------
// Sends the current reading to /api/telemetry. The backend itself appends
// the row to its CSV log on every successful call, so this single request
// both "logs" the reading AND returns the 48h prediction — nothing else
// to wire up on the logging side.
bool sendTelemetryAndGetPrediction() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, skipping telemetry send.");
    return false;
  }

  logHeapStatus("before sync");

  // Drop any socket left over from the previous sync and re-arm TLS. HTTPClient
  // keeps the connection alive by default; Render closes it during the 30s idle
  // gap between syncs, and a WiFiClientSecure that is never stop()ed stays stuck
  // on that dead socket forever - which is why the failures were permanent
  // rather than intermittent, with the heap still looking perfectly healthy.
  secureClient.stop();
  secureClient.setInsecure();

  HTTPClient https;
  if (!https.begin(secureClient, SERVER_URL)) {
    Serial.println("Unable to begin HTTPS connection");
    logHeapStatus("after failed begin()");
    secureClient.stop();
    return false;
  }

  // Render free-tier instances "sleep" after ~15 min idle and can take
  // 20-40s to wake back up on the first request after a gap — give it room.
  https.setTimeout(35000);
  https.setConnectTimeout(35000);
  https.setReuse(false);
  https.addHeader("Content-Type", "application/json");
  https.addHeader("Connection", "close");

  // StaticJsonDocument = stack-allocated, fixed size, no heap churn.
  // 256 bytes comfortably fits 8 numeric/short-string fields.
  StaticJsonDocument<256> reqDoc;
  // Field names below must match the Flask API's `required_fields` exactly.
  // Pressure values are sent as the raw Pascals the Adafruit libraries return
  // (readPressure()), matching whatever the model was trained on for this rig —
  // if predictions look off, check whether the training data used hPa instead
  // and divide by 100 here to match.
  // rain_raw / soil_moisture are sent RAW and uninverted: the backend owns the
  // ADC->mm and ADC->saturation calibration (weather_features.py), so the
  // firmware must not rescale them or the two ends will disagree.
  reqDoc["temperature"]   = dhtTemp;
  reqDoc["humidity"]      = dhtHum;
  reqDoc["temp_bmp180"]   = bmp180Temp;
  reqDoc["pressure_loc1"] = bmp180Pres;
  reqDoc["temp_bmp280"]   = bmp280Temp;
  reqDoc["pressure_loc2"] = bmp280Pres;
  reqDoc["rain_raw"]      = rainAnalog;   // raw ADC, 0-4095, no inversion
  reqDoc["soil_moisture"] = soilAnalog;   // raw ADC, 0-4095, no inversion

  String reqBody;
  serializeJson(reqDoc, reqBody);

  // One immediate retry: if the very first POST after an idle gap still trips
  // over a half-open socket or a cold-started Render instance, the second
  // attempt on a freshly stopped client normally succeeds.
  int httpCode = https.POST(reqBody);
  if (httpCode <= 0) {
    Serial.print("POST failed (code ");
    Serial.print(httpCode);
    Serial.println("), retrying once on a fresh connection...");
    https.end();
    secureClient.stop();
    delay(500);
    secureClient.setInsecure();
    if (https.begin(secureClient, SERVER_URL)) {
      https.setTimeout(35000);
      https.setConnectTimeout(35000);
      https.setReuse(false);
      https.addHeader("Content-Type", "application/json");
      https.addHeader("Connection", "close");
      httpCode = https.POST(reqBody);
    }
  }

  bool success = false;

  if (httpCode == 200 || httpCode == 201) {
    String payload = https.getString();

    // The response also carries the echoed reading, the engineered features and
    // the blend diagnostics, which together far exceed any sane stack buffer.
    // A filter keeps only predictions_48h, so a growing response body can never
    // overflow this document.
    StaticJsonDocument<128> filter;
    filter["predictions_48h"] = true;

    StaticJsonDocument<768> resDoc;
    DeserializationError err = deserializeJson(
      resDoc, payload, DeserializationOption::Filter(filter));

    if (!err) {
      JsonObject pred = resDoc["predictions_48h"];
      if (!pred.isNull()) {
        strlcpy(floodRisk48h, pred["flood_risk_target"] | "", sizeof(floodRisk48h));
        strlcpy(rainCategory48h, pred["rain_category_target"] | "", sizeof(rainCategory48h));
        tempPred48h      = pred["temperature_target"] | 0.0f;
        floodPercent48h  = pred["flood_risk_percent"] | -1.0f;
        rainPercent48h   = pred["rain_percent"] | -1.0f;
        tempPercent48h   = pred["temperature_percent"] | -1.0f;
        havePrediction   = true;
        lastPredictionMillis = millis();
        success = true;

        Serial.print("Prediction OK -> Flood: ");
        if (floodPercent48h >= 0) { Serial.print(floodPercent48h, 0); Serial.print("% "); }
        Serial.print(floodRisk48h);
        Serial.print(" | Rain: ");
        if (rainPercent48h >= 0) { Serial.print(rainPercent48h, 0); Serial.print("% "); }
        Serial.print(rainCategory48h);
        Serial.print(" | Temp(48h): "); Serial.println(tempPred48h, 1);
      } else {
        Serial.println("Response JSON missing predictions_48h object.");
      }
    } else {
      Serial.print("JSON parse error: ");
      Serial.println(err.c_str());
    }
  } else {
    Serial.print("HTTP POST failed, code: ");
    Serial.println(httpCode);
    if (httpCode > 0) {
      Serial.println(https.getString());
    }
  }

  https.end();
  // https.end() alone leaves the TLS socket in whatever state the server left
  // it; stopping the client here guarantees the next sync starts clean.
  secureClient.stop();
  logHeapStatus("after sync");

  // ---- Recovery from repeated sync failures (heap-independent) ----
  if (success) {
    consecutiveFailedSyncs = 0;
  } else {
    consecutiveFailedSyncs++;
    Serial.print("[sync] consecutive failures: ");
    Serial.println(consecutiveFailedSyncs);

    if (consecutiveFailedSyncs == FAILED_SYNCS_BEFORE_WIFI_RESET) {
      Serial.println("[sync] Cycling WiFi to clear a wedged connection...");
      updateStatusLine("WiFi reset...", ST77XX_YELLOW);
      WiFi.disconnect(true);
      delay(500);
      connectWiFi();
    } else if (consecutiveFailedSyncs >= FAILED_SYNCS_BEFORE_RESTART) {
      Serial.println("[sync] Still failing after WiFi reset, restarting...");
      updateStatusLine("Restarting...", ST77XX_RED);
      delay(200);
      ESP.restart();
    }
  }

  // ---- Self-healing restart on sustained heap fragmentation ----
  // Secondary safety net for a genuinely fragmented heap. On this rig it never
  // fires, because maxAllocBlock stays around 110KB; the failed-sync counter
  // above is what recovers the connection failures actually observed.
#if ENABLE_SELF_HEAL_RESTART
  if (ESP.getMaxAllocHeap() < HEAP_FRAGMENTATION_THRESHOLD) {
    consecutiveLowHeapCount++;
    Serial.print("[heap] WARNING: largest free block below threshold (");
    Serial.print(consecutiveLowHeapCount);
    Serial.print("/");
    Serial.print(HEAP_UNHEALTHY_LIMIT);
    Serial.println(" consecutive)");
    if (consecutiveLowHeapCount >= HEAP_UNHEALTHY_LIMIT) {
      Serial.println("[heap] Heap unhealthy for too long, restarting to recover...");
      updateStatusLine("Restarting...", ST77XX_RED);
      delay(200);
      ESP.restart();
    }
  } else {
    consecutiveLowHeapCount = 0;
  }
#endif

  return success;
}

// ---------------- Serial Output ----------------
void printSerialData() {
  Serial.print("Rain_Analog=");   Serial.print(rainAnalog);
  Serial.print(", SMOIST=");      Serial.print(soilAnalog);
  Serial.print(", DHT_Hum=");     Serial.print(dhtHum, 1);
  Serial.print(", DHT_Temp=");    Serial.print(dhtTemp, 1);
  Serial.print(", BMP180_Pres="); Serial.print(bmp180Pres, 1);
  Serial.print(", BMP180_Temp="); Serial.print(bmp180Temp, 1);
  Serial.print(", BMP280_Pres="); Serial.print(bmp280Pres, 1);
  Serial.print(", BMP280_Temp="); Serial.println(bmp280Temp, 1);
}

// ---------------- TFT: Header Banner ----------------
void drawHeader() {
  int16_t x1, y1;
  uint16_t w, h;
  tft.setTextSize(1);
  tft.setTextColor(ST77XX_CYAN, ST77XX_BLACK);
  tft.getTextBounds("WEATHER TELEMETRY", 0, 0, &x1, &y1, &w, &h);
  tft.setCursor((tft.width() - w) / 2, HEADER_Y);
  tft.print("WEATHER TELEMETRY");
  tft.drawFastHLine(0, SEP_Y, tft.width(), 0x39C7); // dim grey divider
}

// ---------------- TFT: Static Labels (drawn once) ----------------
void drawStaticLabels() {
  tft.setTextWrap(false);
  tft.setTextColor(ST77XX_WHITE, ST77XX_BLACK);
  tft.setTextSize(1);

  tft.setCursor(COL1_X, ROW_Y[0]);
  tft.print("RAIN:");
  tft.setCursor(COL2_X, ROW_Y[0]);
  tft.print("SMOI:");

  tft.setCursor(COL1_X, ROW_Y[1]);
  tft.print("DHUM:");
  tft.setCursor(COL2_X, ROW_Y[1]);
  tft.print("DTMP:");

  tft.setCursor(COL1_X, ROW_Y[2]);
  tft.print("B180P:");
  tft.setCursor(COL2_X, ROW_Y[2]);
  tft.print("B180T:");

  tft.setCursor(COL1_X, ROW_Y[3]);
  tft.print("B280P:");
  tft.setCursor(COL2_X, ROW_Y[3]);
  tft.print("B280T:");
}

// ---------------- TFT: Bottom Half — 48h Gauges ----------------
void drawGaugeTitles() {
  int16_t x1, y1;
  uint16_t w, h;
  const char* titles[3] = {"FLOOD", "RAIN", "TEMP"};

  tft.setTextSize(1);
  tft.setTextColor(ST77XX_WHITE, ST77XX_BLACK);
  for (int i = 0; i < 3; i++) {
    tft.getTextBounds(titles[i], 0, 0, &x1, &y1, &w, &h);
    tft.setCursor(GAUGE_CX[i] - w / 2, GAUGE_TITLE_Y);
    tft.print(titles[i]);
  }

  for (int i = 0; i < 3; i++) {
    drawGaugeTrack(GAUGE_CX[i], GAUGE_CY, GAUGE_R);
    drawNeedle(GAUGE_CX[i], GAUGE_CY, GAUGE_R, 0.5, ST77XX_WHITE);
  }
}

void drawGaugeTrack(int cx, int cy, int r) {
  for (int a = 0; a <= 180; a += 3) {
    float rad = a * PI / 180.0;
    int x1 = cx + (int)((r - 4) * cos(rad));
    int y1 = cy - (int)((r - 4) * sin(rad));
    int x2 = cx + (int)(r * cos(rad));
    int y2 = cy - (int)(r * sin(rad));

    uint16_t col;
    if (a >= 120)      col = ST77XX_GREEN;
    else if (a >= 60)  col = ST77XX_YELLOW;
    else               col = ST77XX_RED;

    tft.drawLine(x1, y1, x2, y2, col);
  }
  tft.drawFastHLine(cx - r, cy, r * 2, 0x39C7);
}

void drawNeedle(int cx, int cy, int r, float value01, uint16_t color) {
  if (value01 < 0) value01 = 0;
  if (value01 > 1) value01 = 1;

  tft.fillCircle(cx, cy, r - 5, ST77XX_BLACK);
  tft.fillCircle(cx, cy, 2, ST77XX_WHITE);

  float angleDeg = 180.0 - (value01 * 180.0);
  float rad = angleDeg * PI / 180.0;
  int tipX = cx + (int)((r - 6) * cos(rad));
  int tipY = cy - (int)((r - 6) * sin(rad));

  tft.drawLine(cx, cy, tipX, tipY, color);
  tft.drawLine(cx + 1, cy, tipX + 1, tipY, color);
  tft.drawLine(cx, cy - 1, tipX, tipY - 1, color);
}

// Case-insensitive substring check without pulling in Arduino String —
// small local helper so categoryToValue01 can work on plain char buffers.
static bool containsIgnoreCase(const char* haystack, const char* needle) {
  if (!haystack || !needle) return false;
  size_t hLen = strlen(haystack);
  size_t nLen = strlen(needle);
  if (nLen == 0 || nLen > hLen) return false;
  for (size_t i = 0; i + nLen <= hLen; i++) {
    size_t j = 0;
    for (; j < nLen; j++) {
      if (tolower((unsigned char)haystack[i + j]) != tolower((unsigned char)needle[j])) break;
    }
    if (j == nLen) return true;
  }
  return false;
}

// Maps a category keyword to a 0.0-1.0 gauge position. Only used as a fallback
// when the backend does not send the 0-100 percentages.
float categoryToValue01(const char* s) {
  if (containsIgnoreCase(s, "high") || containsIgnoreCase(s, "severe") || containsIgnoreCase(s, "heavy")) return 0.85;
  if (containsIgnoreCase(s, "moderate") || containsIgnoreCase(s, "medium")) return 0.5;
  return 0.15; // low / light / none / default
}

void drawGauge(int idx, float value01, const char* valueText, uint16_t valueColor) {
  drawNeedle(GAUGE_CX[idx], GAUGE_CY, GAUGE_R, value01, ST77XX_WHITE);
  drawValueBoxW(GAUGE_CX[idx] - GAUGE_VAL_W / 2, GAUGE_VAL_Y, GAUGE_VAL_W, VAL_H, valueText, valueColor);
}

// ---------------- TFT: Update Value Boxes Only ----------------
void drawValueBoxW(int x, int y, int w, int h, const char* text, uint16_t color) {
  tft.fillRect(x, y, w, h, ST77XX_BLACK);
  tft.setTextColor(color, ST77XX_BLACK);
  tft.setCursor(x, y);
  tft.print(text);
}

void drawValueBox(int x, int y, const char* text, uint16_t color) {
  drawValueBoxW(x, y, VAL_W, VAL_H, text, color);
}

void updateDisplayValues() {
  char buf[16];

  snprintf(buf, sizeof(buf), "%d", rainAnalog);
  drawValueBox(COL1_X + 32, ROW_Y[0], buf, ST77XX_CYAN);

  snprintf(buf, sizeof(buf), "%d", soilAnalog);
  drawValueBox(COL2_X + 32, ROW_Y[0], buf, ST77XX_CYAN);

  snprintf(buf, sizeof(buf), "%.1f%%", dhtHum);
  drawValueBox(COL1_X + 38, ROW_Y[1], buf, ST77XX_WHITE);

  snprintf(buf, sizeof(buf), "%.1fC", dhtTemp);
  drawValueBox(COL2_X + 38, ROW_Y[1], buf, ST77XX_WHITE);

  snprintf(buf, sizeof(buf), "%.0f", bmp180Pres);
  drawValueBox(COL1_X + 38, ROW_Y[2], buf, ST77XX_ORANGE);

  snprintf(buf, sizeof(buf), "%.1fC", bmp180Temp);
  drawValueBox(COL2_X + 38, ROW_Y[2], buf, ST77XX_ORANGE);

  snprintf(buf, sizeof(buf), "%.0f", bmp280Pres);
  drawValueBox(COL1_X + 38, ROW_Y[3], buf, ST77XX_MAGENTA);

  snprintf(buf, sizeof(buf), "%.1fC", bmp280Temp);
  drawValueBox(COL2_X + 38, ROW_Y[3], buf, ST77XX_MAGENTA);
}

// Classifies the 48h predicted temperature vs the current DHT11 reading
// into a simple "overheat tendency" label. Returns an enum instead of a
// String — text/colour mapping happens in updateGauges().
TempTendency computeTempTendency() {
  if (!havePrediction) return TEND_UNKNOWN;
  float diff = tempPred48h - dhtTemp;
  if (diff >= 1.0) return TEND_UP;
  if (diff <= -1.0) return TEND_DOWN;
  return TEND_STABLE;
}

void updateGauges() {
  char buf[24];

  if (!havePrediction) {
    drawGauge(0, 0.5, "...", ST77XX_WHITE);
    drawGauge(1, 0.5, "...", ST77XX_WHITE);
    drawGauge(2, 0.5, "...", ST77XX_WHITE);
    return;
  }

  // Gauge 0 — Flood risk. Needle follows the 0-100 score when the backend
  // supplies one; otherwise it falls back to the coarse category position.
  float floodVal = (floodPercent48h >= 0) ? floodPercent48h / 100.0
                                          : categoryToValue01(floodRisk48h);
  uint16_t floodColor = (floodVal >= 0.6) ? ST77XX_RED : (floodVal >= 0.25 ? ST77XX_YELLOW : ST77XX_GREEN);
  if (floodPercent48h >= 0) {
    snprintf(buf, sizeof(buf), "%.0f%% %s", floodPercent48h, floodRisk48h);
    drawGauge(0, floodVal, buf, floodColor);
  } else {
    drawGauge(0, floodVal, floodRisk48h, floodColor);
  }

  // Gauge 1 — Rainfall tendency
  float rainVal = (rainPercent48h >= 0) ? rainPercent48h / 100.0
                                        : categoryToValue01(rainCategory48h);
  uint16_t rainColor = (rainVal >= 0.7) ? ST77XX_RED : (rainVal >= 0.45 ? ST77XX_ORANGE : ST77XX_CYAN);
  if (rainPercent48h >= 0) {
    snprintf(buf, sizeof(buf), "%.0f%% %s", rainPercent48h, rainCategory48h);
    drawGauge(1, rainVal, buf, rainColor);
  } else {
    drawGauge(1, rainVal, rainCategory48h, rainColor);
  }

  // Gauge 2 — Heat tendency. The percentage is a position on a 15-45 C scale,
  // so the needle shows how hot it will be while the label keeps the actual
  // forecast temperature and the colour keeps the up/down/stable trend.
  TempTendency tend = computeTempTendency();
  float tempVal = (tempPercent48h >= 0)
                    ? tempPercent48h / 100.0
                    : ((tend == TEND_UP) ? 0.85 : (tend == TEND_DOWN ? 0.15 : 0.5));
  uint16_t tempColor = (tend == TEND_UP) ? ST77XX_RED : (tend == TEND_DOWN ? ST77XX_CYAN : ST77XX_WHITE);
  snprintf(buf, sizeof(buf), "%.1fC", tempPred48h);
  drawGauge(2, tempVal, buf, tempColor);
}

void updateStatusLine(const char* text, uint16_t color) {
  drawValueBoxW(COL1_X, STATUS_Y, STATUS_W, VAL_H, text, color);
}

////////////////////////////////////////////////////////////////////