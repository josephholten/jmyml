#include <raylib.h>
#include <raymath.h>
#include <fmt/ranges.h>
#include <vector>

#ifndef Real
#define Real float
#endif

int main() {
    const int initScreenWidth = 400;
    const int initScreenHeight = 400;

    const int screenWidth = initScreenWidth;
    const int screenHeight = initScreenHeight;

    Color bg_color = BLACK;
    Color fg_color = WHITE;

    InitWindow(screenWidth, screenHeight, "mldraw");
    SetTargetFPS(60);

    std::vector<std::vector<Vector2>> lines;
    bool down = false;
    bool takingImage = false;

    char detected = '_';
    float confidence = 0.0;
    char text[] = "detected: _ (0.00)";

    // TODO
    // Model model = jmyml::nn::load("models/mnist-good.model");

    while (!WindowShouldClose()) {
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            if (down == false) {
                down = true;
                lines.push_back({GetMousePosition()});
            } else {
                lines.back().push_back(GetMousePosition());
            }
        } else {
            down = false;
        }

        if (IsKeyPressed(KEY_C)) {
            takingImage = true;
        }

        if (IsKeyPressed(KEY_ENTER)) {
            takingImage = true;
        }

        BeginDrawing(); {
            if (!takingImage) {
                DrawText("please draw a digit, enter to detect, 'c' to clear, 'ESC' to quit", 0, 0, 22, GRAY);
                sprintf(text, "detected: %c (%.2f)", detected, confidence);
                DrawText(text, 0, 25, 22, GRAY);
            }
            for (auto line : lines) {
                DrawSplineCatmullRom(line.data(), line.size(), 5., fg_color);
            }
            ClearBackground(bg_color);
            if (takingImage) {
                Image image = LoadImageFromScreen();
                TraceLog(LOG_INFO, "loaded image %dx%d from screen", image.height, image.width);
                if (image.format != PIXELFORMAT_UNCOMPRESSED_R8G8B8A8) {
                    TraceLog(LOG_ERROR, "image taken from screen is in unrecognized format");
                } else {
                    // 1. step: convert to PIXEL_FORMAT_UNCOMPRESSED_GRAYSCALE
                    ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);

                    // 2. step: crop to content (square)
                    // TODO

                    // 3. step: scale to 28x28
                    // TODO

                    // 4. step: send it through NN
                    // std::vector<Real> preds = Model.forward(image.data, image.width*image.height);
                    // auto max_it = std::max_element(preds.begin(), preds.end());
                    // detected = max_it - preds.begin() + '0';
                    // confidence = *max_it;
                    detected = '?';
                    TraceLog(LOG_ERROR, "detection unimplemented");
                }
                takingImage = false;
            }
        } EndDrawing();
    }

    CloseWindow();
    return 0;
}