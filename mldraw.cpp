#include <raylib.h>
#include <raymath.h>
#include <fmt/ranges.h>
#include <vector>

void drawInterface() {

}

int main() {
    const int initScreenWidth = 800;
    const int initScreenHeight = 450;

    const int screenWidth = initScreenWidth;
    const int screenHeight = initScreenHeight;

    Color bg_color = BLACK;
    Color fg_color = WHITE;

    InitWindow(screenWidth, screenHeight, "mldraw");
    SetTargetFPS(60);

    // Image drawing = GenImageColor(screenWidth, screenHeight, bg_color);
    // Texture texture;

    std::vector<std::vector<Vector2>> lines;
    bool down = false;

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


        BeginDrawing(); {
            for (auto line : lines) {
                DrawLineStrip(line.data(), line.size(), fg_color);
            }
            ClearBackground(bg_color);
        } EndDrawing();
    }

    CloseWindow();
    return 0;
}