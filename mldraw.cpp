#include <raylib.h>
#include <raymath.h>
#include <fmt/ranges.h>

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

    Image drawing = GenImageColor(screenWidth, screenHeight, bg_color);
    Texture texture;

    while (!WindowShouldClose()) {
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 mouse_end = GetMousePosition();
            Vector2 mouse_start = Vector2Subtract(mouse_end, GetMouseDelta());

            ImageDrawLineEx(&drawing, mouse_start, mouse_end, 2., fg_color);
            UnloadTexture(texture);
            texture = LoadTextureFromImage(drawing);
        }

        BeginDrawing(); {
            DrawTexture(texture, 0, 0, fg_color);
            ClearBackground(bg_color);
        } EndDrawing();
    }

    CloseWindow();
    return 0;
}