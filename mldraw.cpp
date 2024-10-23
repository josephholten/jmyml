#include <raylib.h>
#include <raymath.h>
#include <fmt/ranges.h>
#include <vector>
#include <array>
#include <cassert>

#ifndef Real
#define Real float
#endif

// first,last line,column containing non zero entry
// NOTE: assuming PIXELFORMAT_UNCOMPRESSED_GRAYSCALE
Rectangle get_crop_rect(Image image, Vector2* points) {
    Rectangle rect;
    uint8_t* data = (uint8_t*)image.data;

    for (int row = 0; row < image.height; row++) {
        for (int col = 0; col < image.width; col++) {
            if (data[row*image.width + col]) {
                rect.y = (float)row;
                points[0].x = (float)col;
                points[0].y = (float)row;
                goto x;
            }
        }
    }
    goto zero;

x:
    for (int col = 0; col < image.width; col++) {
        for (int row = 0; row < image.height; row++) {
            if (data[row*image.width + col]) {
                rect.x = (float)col;
                points[1].x = (float)col;
                points[1].y = (float)row;
                goto width;
            }
        }
    }
    goto zero;

width:
    for (int row = image.height-1; row > rect.y; row--) {
        for (int col = 0; col < image.width; col++) {
            if (data[row*image.width + col]) {
                points[2].x = (float)col;
                points[2].y = (float)row;
                rect.height = row+1 - rect.y;
                goto height;
            }
        }
    }
    rect.height = image.height+1 - rect.y;

height:
    for (int col = image.width-1; col > rect.x; col--) {
        for (int row = 0; row < image.height; row++) {
            if (data[row*image.width + col]) {
                points[3].x = (float)col;
                points[3].y = (float)row;
                rect.width = col+1 - rect.x;
                goto end;
            }
        }
    }
    rect.width = image.width+1 - rect.x;

zero:
    rect.y = image.height;
    rect.x = image.width;
    rect.height = 0;
    rect.width = 0;

end:
    assert(rect.x >= 0 && rect.y >= 0 && rect.width >= 0 && rect.height >= 0);
    return rect;
}

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
    int detectionCount = 0;
    Rectangle crop = {.x = 0, .y = 0, .width = screenWidth, .height = screenHeight};
    std::array<Vector2, 4> ps;

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
            lines.clear();
        }

        if (IsKeyPressed(KEY_ENTER)) {
            takingImage = true;
        }

        BeginDrawing(); {
            ClearBackground(bg_color);
            if (!takingImage) {
                DrawText("please draw a digit, enter to detect, 'c' to clear, 'ESC' to quit", 0, 0, 22, GRAY);
                sprintf(text, "detected: %c (%.2f)", detected, confidence);
                DrawText(text, 0, 25, 22, GRAY);
                if (detected != '_')
                    DrawRectangleLinesEx(crop, 2., MAGENTA);
                for (auto p : ps)
                    DrawCircleV(p, 3., GREEN);
            }

            for (auto line : lines) {
                DrawSplineCatmullRom(line.data(), line.size(), 5., fg_color);
            }
        } EndDrawing();

        if (takingImage) {
            Image image = LoadImageFromScreen();
            TraceLog(LOG_INFO, "loaded image %dx%d from screen", image.height, image.width);
            if (image.format != PIXELFORMAT_UNCOMPRESSED_R8G8B8A8) {
                TraceLog(LOG_ERROR, "image taken from screen is in unrecognized format");
            } else {
                // 1. step: convert to PIXEL_FORMAT_UNCOMPRESSED_GRAYSCALE
                ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);

                // 2. step: crop to content
                // TODO: crop in square aspect ratio
                crop = get_crop_rect(image, ps.data());
                ImageCrop(&image, crop);
                TraceLog(LOG_INFO, "cropped image at %.1fx%.1f to size %.1fx%.1f", crop.x, crop.y, crop.width, crop.height);

                // 3. step: scale to 28x28
                ImageResize(&image, 28, 28);

                // 4. step: send it through NN
                // std::vector<Real> preds = Model.forward(image.data, image.width*image.height);
                // auto max_it = std::max_element(preds.begin(), preds.end());
                // detected = max_it - preds.begin() + '0';
                // confidence = *max_it;
                detected = '?';
                TraceLog(LOG_ERROR, "detection unimplemented");

                detectionCount++;
                char file_name[] = "detection???.png";
                sprintf(file_name, "detection%03d.png", detectionCount);
                ExportImage(image, file_name);
                TraceLog(LOG_INFO, "exported image as");
            }
            takingImage = false;
        }
    }

    CloseWindow();
    return 0;
}