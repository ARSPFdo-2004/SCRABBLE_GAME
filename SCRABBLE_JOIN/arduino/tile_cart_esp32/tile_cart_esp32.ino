#include <WiFi.h>
#include <WebServer.h>
#if __has_include(<esp_arduino_version.h>)
#include <esp_arduino_version.h>
#endif

#ifndef ESP_ARDUINO_VERSION_MAJOR
#define ESP_ARDUINO_VERSION_MAJOR 2
#endif

// ================= WIFI =================
const char* ssid = "YOUR_HOTSPOT_NAME";
const char* password = "YOUR_HOTSPOT_PASSWORD";

WebServer server(80);

// ================= L298N =================
#define IN1_CAR 12
#define IN2_CAR 14
#define IN3_CAR 27
#define IN4_CAR 26

#define ENA_CAR 18
#define ENB_CAR 5

// ================= PWM =================
#define PWM_FREQ 1000
#define PWM_RESOLUTION 8
#define PWM_CHANNEL_LEFT 0
#define PWM_CHANNEL_RIGHT 1

int leftSpeed = 200;
int rightSpeed = 200;

float moveDistanceCm = 5.0;
// Initial calibration: 130 ms/cm at PWM 35. Movement time is adjusted when
// the GUI changes PWM so the requested distance remains approximately stable.
const float CALIBRATION_PWM = 35.0;
const float MILLISECONDS_PER_CM_AT_CALIBRATION_PWM = 130.0;

// ================= FUNCTION PROTOTYPES =================
void forward();
void backward();
void stopCar();
void moveForward();
void moveBackward();
void handleRoot();
void handleSpeed();
void handleDistance();
void applyMovementSettings();
unsigned long movementTimeMs();
String formatDistance();

void setupMotorPwm() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(ENA_CAR, PWM_FREQ, PWM_RESOLUTION);
  ledcAttach(ENB_CAR, PWM_FREQ, PWM_RESOLUTION);
#else
  ledcSetup(PWM_CHANNEL_LEFT, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(PWM_CHANNEL_RIGHT, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(ENA_CAR, PWM_CHANNEL_LEFT);
  ledcAttachPin(ENB_CAR, PWM_CHANNEL_RIGHT);
#endif
}

void writeMotorPwm(int leftPwm, int rightPwm) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(ENA_CAR, leftPwm);
  ledcWrite(ENB_CAR, rightPwm);
#else
  ledcWrite(PWM_CHANNEL_LEFT, leftPwm);
  ledcWrite(PWM_CHANNEL_RIGHT, rightPwm);
#endif
}

void setup() {
  Serial.begin(115200);

  pinMode(IN1_CAR, OUTPUT);
  pinMode(IN2_CAR, OUTPUT);
  pinMode(IN3_CAR, OUTPUT);
  pinMode(IN4_CAR, OUTPUT);

  setupMotorPwm();

  stopCar();

  // ================= CONNECT TO HOTSPOT =================
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.println();
  Serial.print("Connecting to hotspot");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("=================================");
  Serial.println("ESP32 CONNECTED");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.println("Open this IP in your browser");
  Serial.println("=================================");

  // ================= WEB ROUTES =================
  server.on("/", handleRoot);

  server.on("/forward", []() {
    applyMovementSettings();
    moveForward();
    String message = "Moved Forward " + formatDistance() + " cm at PWM " + String(leftSpeed);
    server.send(200, "text/plain", message);
  });

  server.on("/backward", []() {
    applyMovementSettings();
    moveBackward();
    String message = "Moved Backward " + formatDistance() + " cm at PWM " + String(leftSpeed);
    server.send(200, "text/plain", message);
  });

  server.on("/stop", []() {
    stopCar();
    server.send(200, "text/plain", "Stopped");
  });

  server.on("/speed", handleSpeed);
  server.on("/distance", handleDistance);

  server.onNotFound([]() {
    server.send(404, "text/plain", "Route not found");
  });

  server.begin();
  Serial.println("Web Server Started");
}

void loop() {
  server.handleClient();
}

// ================= SETTINGS =================
void handleSpeed() {
  if (!server.hasArg("val")) {
    server.send(400, "text/plain", "Missing speed value");
    return;
  }

  int requestedSpeed = server.arg("val").toInt();
  if (requestedSpeed < 1 || requestedSpeed > 255) {
    server.send(400, "text/plain", "PWM speed must be between 1 and 255");
    return;
  }

  leftSpeed = requestedSpeed;
  rightSpeed = requestedSpeed;
  server.send(200, "text/plain", "PWM speed set to " + String(requestedSpeed));
}

void handleDistance() {
  if (!server.hasArg("val")) {
    server.send(400, "text/plain", "Missing distance value");
    return;
  }

  float requestedDistance = server.arg("val").toFloat();
  if (requestedDistance <= 0.0) {
    server.send(400, "text/plain", "Distance must be greater than 0 cm");
    return;
  }

  moveDistanceCm = requestedDistance;
  server.send(200, "text/plain", "Distance set to " + formatDistance() + " cm");
}

void applyMovementSettings() {
  if (server.hasArg("speed")) {
    int requestedSpeed = server.arg("speed").toInt();
    if (requestedSpeed >= 1 && requestedSpeed <= 255) {
      leftSpeed = requestedSpeed;
      rightSpeed = requestedSpeed;
    }
  }

  if (server.hasArg("distance")) {
    float requestedDistance = server.arg("distance").toFloat();
    if (requestedDistance > 0.0) {
      moveDistanceCm = requestedDistance;
    }
  }
}

// ================= WEB PAGE =================
void handleRoot() {
  String distance = formatDistance();
  String html =
    "<!DOCTYPE html>"
    "<html>"
    "<head>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    "<title>ESP32 Car</title>"
    "<script>"
    "function sendCommand(cmd){"
    "fetch('/' + cmd)"
    ".then(res => res.text())"
    ".then(data => {document.getElementById('status').innerHTML = data;});"
    "}"
    "</script>"
    "</head>"
    "<body style='text-align:center;font-family:Arial;'>"
    "<h2>ESP32 CAR CONTROL</h2>"
    "<p id='status'>Ready - PWM " + String(leftSpeed) + ", " + distance + " cm</p>"
    "<button onclick=\"sendCommand('forward')\" style='width:220px;height:70px;font-size:22px;'>"
    "FORWARD " + distance + " cm"
    "</button>"
    "<br><br>"
    "<button onclick=\"sendCommand('backward')\" style='width:220px;height:70px;font-size:22px;'>"
    "BACKWARD " + distance + " cm"
    "</button>"
    "<br><br>"
    "<button onclick=\"sendCommand('stop')\" style='width:220px;height:70px;font-size:22px;'>"
    "STOP"
    "</button>"
    "</body>"
    "</html>";

  server.send(200, "text/html", html);
}

// ================= MOVEMENT =================
unsigned long movementTimeMs() {
  if (leftSpeed <= 0) {
    return 0;
  }
  float speedScale = CALIBRATION_PWM / (float)leftSpeed;
  return (unsigned long)(
    moveDistanceCm * MILLISECONDS_PER_CM_AT_CALIBRATION_PWM * speedScale
  );
}

String formatDistance() {
  if (moveDistanceCm == (int)moveDistanceCm) {
    return String((int)moveDistanceCm);
  }
  return String(moveDistanceCm, 1);
}

void moveForward() {
  forward();
  delay(movementTimeMs());
  stopCar();
}

void moveBackward() {
  backward();
  delay(movementTimeMs());
  stopCar();
}

// ================= MOTOR CONTROL =================
void forward() {
  digitalWrite(IN1_CAR, HIGH);
  digitalWrite(IN2_CAR, LOW);
  digitalWrite(IN3_CAR, HIGH);
  digitalWrite(IN4_CAR, LOW);

  writeMotorPwm(leftSpeed, rightSpeed);
}

void backward() {
  digitalWrite(IN1_CAR, LOW);
  digitalWrite(IN2_CAR, HIGH);
  digitalWrite(IN3_CAR, LOW);
  digitalWrite(IN4_CAR, HIGH);

  writeMotorPwm(leftSpeed, rightSpeed);
}

void stopCar() {
  digitalWrite(IN1_CAR, LOW);
  digitalWrite(IN2_CAR, LOW);
  digitalWrite(IN3_CAR, LOW);
  digitalWrite(IN4_CAR, LOW);

  writeMotorPwm(0, 0);
}
