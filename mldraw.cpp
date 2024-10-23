#include <raylib.h>
#include <raymath.h>
#include <fmt/ranges.h>
#include <vector>

void drawInterface() {

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
    char text[] = "detected: _";

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
                sprintf(text, "detected: %c", detected);
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
                    TraceLog(LOG_ERROR, "detection unimplemented");
                    // TODO:
                    //   - convert to PIXEL_FORMAT_UNCOMPRESSED_GRAYSCALE
                    //   - crop to content (square)
                    //   - scale it to 28x28
                    //   - send it through NN
                    //   - display prediction from NN
                }
                takingImage = false;
            }
        } EndDrawing();
    }

    CloseWindow();
    return 0;
}