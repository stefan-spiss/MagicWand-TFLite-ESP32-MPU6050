/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "accelerometer_handler.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

int begin_index = 0;

Adafruit_MPU6050 mpu;

float save_data[600] = {0.0};
bool pending_initial_data = true;
long last_sample_millis = 0;

TfLiteStatus SetupAccelerometer(tflite::ErrorReporter* error_reporter) {
  // Try to initialize!
  if (!mpu.begin()) {
    while (1) {
      error_reporter->Report("IMU not found");
      delay(10);
    }
  }
  error_reporter->Report("IMU found");
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  error_reporter->Report("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    error_reporter->Report("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    error_reporter->Report("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    error_reporter->Report("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    error_reporter->Report("+-16G");
    break;
  }
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  error_reporter->Report("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    error_reporter->Report("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    error_reporter->Report("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    error_reporter->Report("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    error_reporter->Report("+- 2000 deg/s");
    break;
  }
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  error_reporter->Report("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    error_reporter->Report("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    error_reporter->Report("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    error_reporter->Report("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    error_reporter->Report("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    error_reporter->Report("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    error_reporter->Report("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    error_reporter->Report("5 Hz");
    break;
  }
  
  return kTfLiteOk;
}

static bool UpdateData() {
  bool new_data = false;
  if ((millis() - last_sample_millis) < 40) {
    return false;
  }
  last_sample_millis = millis();

  float accX = 0.0F;
  float accY = 0.0F;
  float accZ = 0.0F;
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  accX = a.acceleration.x;
  accY = a.acceleration.y;
  accZ = a.acceleration.z;

  /* save_data[begin_index++] = 1000 * accX; */
  /* save_data[begin_index++] = 1000 * accY; */
  /* save_data[begin_index++] = 1000 * accZ; */
  /* save_data[begin_index++] = 100.0f * accX; */
  /* save_data[begin_index++] = 100.0f * accY; */
  /* save_data[begin_index++] = 100.0f * accZ; */
  save_data[begin_index++] = accX;
  save_data[begin_index++] = accY;
  save_data[begin_index++] = accZ;

  if (begin_index >= 600) {
    begin_index = 0;
  }
  new_data = true;

  return new_data;
}

bool ReadAccelerometer(tflite::ErrorReporter* error_reporter, float* input,
                       int length, bool reset_buffer) {
   if (reset_buffer) {
    memset(save_data, 0, 600 * sizeof(float));
    begin_index = 0;
    pending_initial_data = true;
  }

  if (!UpdateData()) {
    return false;
  }

  if (pending_initial_data && begin_index >= 200) {
    pending_initial_data = false;
  }

  if (pending_initial_data) {
    return false;
  }

  for (int i = 0; i < length; ++i) {
    int ring_array_index = begin_index + i - length;
    if (ring_array_index < 0) {
      ring_array_index += 600;
    }
    input[i] = save_data[ring_array_index];
  }
  return true;
}
