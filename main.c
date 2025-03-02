/* Sampling done take average and control servo and LED*/
if (i >= RESULT_SIZE) {
  i = 0;
  
  // Calculate average temperature from ADC readings
  tmpValue = 0;
  ldrValue = 0;
  for(uint16_t j = 0; j < RESULT_SIZE; j++) {
    tmpValue += gAdcResult0[j];
    ldrValue += gAdcResult1[j];
  }
  tmpValue /= RESULT_SIZE;
  ldrValue /= RESULT_SIZE;
  
  // Convert ADC value to temperature
  adcVolt_TMP = (tmpValue * 3.3f) / 4096.0f;
  temperatureC = ((adcVolt_TMP - 0.5f) * 100.0f);
  
  // Control LED brightness based on light conditions
  if(ldrValue < COVERED) {  // Dark condition
    DL_TimerG_setCaptureCompareValue(PWM_LED_INST, BRIGHT, DL_TIMER_CC_1_INDEX);
    // Force vent to close in dark conditions
    if(vent_state != 0) {
      shut_vent();
      vent_state = 0;
    }
  }
  else if(ldrValue < SHADED) {  // Dim condition
    DL_TimerG_setCaptureCompareValue(PWM_LED_INST, DIM, DL_TIMER_CC_1_INDEX);
    // Control vent based on temperature
    if(temperatureC >= 25.0f) {
      if(vent_state == 0) {
        open_vent();
        vent_state = 1;
      }
    } else {
      if(vent_state == 1) {
        shut_vent();
        vent_state = 0;
      }
    }
  }
  else {  // Well-lit condition
    DL_TimerG_setCaptureCompareValue(PWM_LED_INST, OFF, DL_TIMER_CC_1_INDEX);
    // Control vent based on temperature
    if(temperatureC >= 25.0f) {
      if(vent_state == 0) {
        open_vent();
        vent_state = 1;
      }
    } else {
      if(vent_state == 1) {
        shut_vent();
        vent_state = 0;
      }
    }
  }
  
  delay_cycles(SAMPLE_PERIOD);
} 