#include "NeuralNetwork.h"
#include "selected_data.h"
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_timer.h"  // 添加计时器头文件
#include "esp_task_wdt.h" 

static const char *TAG = "MAIN";
static const char *gesture_names[6] = {
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
};

extern "C" void app_main(void)
{
    // 设置看门狗
    // esp_task_wdt_config_t wdt_config = {
    //     .timeout_ms = 10000,  // 10秒
    //     .idle_core_mask = (1 << portNUM_PROCESSORS) - 1,    // 所有核心
    //     .trigger_panic = false
    // };
    // ESP_ERROR_CHECK(esp_task_wdt_init(&wdt_config));

    // 增加当前任务的栈大小
    vTaskPrioritySet(NULL, tskIDLE_PRIORITY + 2);
    
    esp_log_level_set("*", ESP_LOG_INFO);
    ESP_LOGI(TAG, "Starting Neural Network...");

    // Create an instance of the neural network
    NeuralNetwork *nn = new NeuralNetwork();

    int numSamples = NUM_SAMPLES;
    int correctCount = 0;

    // TODO: Add a timer to record the inference time (13 pts)
    // YOUR CODE HERE

    // 添加计时器相关变量
    int64_t start_time = 0;
    int64_t end_time = 0;
    int64_t total_inference_time = 0;

    for (int i = 0; i < numSamples; i++) {
        float *inputBuffer = nn->getInputBuffer();
        for (int j = 0; j < FEATURE_SIZE; j++) {
            inputBuffer[j] = X_selected[i][j];
        }

        // 开始计时
        start_time = esp_timer_get_time();
        
        nn->predict();
         
        // 结束计时
        end_time = esp_timer_get_time();
        
        // 累加推理时间（微秒）
         total_inference_time += (end_time - start_time);

        float *outputBuffer = nn->getOutputBuffer();
        int predictedLabel = 0;
        float maxProb = outputBuffer[0];
        for (int k = 1; k < 6; k++) {
            if (outputBuffer[k] > maxProb) {
                maxProb = outputBuffer[k];
                predictedLabel = k;
            }
        }

        int trueLabel = y_selected[i];
        if (predictedLabel == trueLabel) {
            correctCount++;
        }

        ESP_LOGI(TAG, "Sample %d: GT=%d, GT_gesture=%s, Predicted=%d, Predicted_gesture=%s",
                 i, trueLabel, gesture_names[trueLabel],
                 predictedLabel, gesture_names[predictedLabel]);
    }

    
    float avg_inference_time = (float)total_inference_time / numSamples;
    ESP_LOGI(TAG, "Average inference time: %.2f ms", avg_inference_time / 1000.0);  // 转换为毫秒

    float accuracy = (float)correctCount / numSamples;
    ESP_LOGI(TAG, "Final Accuracy: %.5f%%", accuracy * 100);

    // END OF YOUR CODE
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}