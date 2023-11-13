// Basic demo for accelerometer readings from Adafruit MPU6050

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;
long last_sample_millis = 0;
bool capture = false;
char a;

void setup(void) {
  Serial.begin(115200);
  while (!Serial) {
    delay(10); // will pause Zero, Leonardo, etc until serial console opens
  }

  // Try to initialize!
  while (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    delay(10);
    //while (1) {
      //delay(10);
    //}
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("");
  delay(100);
}

void capture_data() {
  /* Get new sensor events with the readings */
  if ((millis() - last_sample_millis) >= 40) {
    
    last_sample_millis = millis();
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
  
    /* Print out the values */
    Serial.print(a.acceleration.x);
    Serial.print(",");
    Serial.print(a.acceleration.y);
    Serial.print(",");
    Serial.print(a.acceleration.z);
    Serial.print("\n");
  }
}

void loop() {
  if (Serial.available() > 0) {
    a = Serial.read();
    if (a == 'o') {
      Serial.print("-,-,-\n");
      capture = true;
    } else if (a == 'p') {
      capture = false;
      Serial.print("\n\n\n\n");
    }
  }
  if (capture) {
    capture_data();
  }
}
