#include <cassert>
#include <jmyml/data/IdxFile.hpp>
#include <raylib.h>

static int roundDown(int x, int multiple) {
    return (x / multiple) * multiple;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fmt::println("pass exactly one path to an idx file");
        exit(1);
    }

    char* path = argv[1];

    jmyml::IdxFile idx;
    idx.load(path);

    const int initScreenWidth = 800;
    const int initScreenHeight = 450;

    const int imageWidth = idx.shape[idx.shape.size()-1];
    const int imageHeight = idx.shape[idx.shape.size()-2];
    const int imageSize = imageWidth*imageHeight;

    const int imageColumns = initScreenWidth / imageWidth;
    const int imageRows = initScreenHeight / imageHeight;

    const int screenWidth = imageColumns * imageWidth;
    const int screenHeight = imageRows * imageHeight;

    fmt::println("images:");
    fmt::println("  rows: {}", imageRows);
    fmt::println("  cols: {}", imageColumns);

    int page = 0;

    InitWindow(screenWidth, screenHeight, "idxview");
    SetTargetFPS(60);

    std::vector<Texture> textures(imageColumns*imageRows);

    auto load_page = [&](int page){
        for (int row = 0; row < imageRows; row++) {
            for (int col = 0; col < imageColumns; col++) {
                // Create Raylib image from raw pixel data
                Image image = {
                    .data = (uint8_t*)idx.data + page * screenHeight*screenWidth + row * imageColumns*imageSize + col * imageSize,
                    .width = imageWidth,
                    .height = imageHeight,
                    .mipmaps = 1,
                    .format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
                };
                // Load texture from the image
                textures[row*imageColumns+col] = LoadTextureFromImage(image);
            }
        }
    };

    load_page(page);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_RIGHT)) {
            page++;
            load_page(page);
        } else if (IsKeyPressed(KEY_LEFT) && page > 0) {
            page--;
            load_page(page);
        }

        BeginDrawing(); {
            ClearBackground(BLACK);
            for (int row = 0; row < imageRows; row++) {
                for (int col = 0; col < imageColumns; col++) {
                    DrawTexture(textures[row*imageColumns+col], col*imageWidth, row*imageHeight, WHITE);
                }
            }
        } EndDrawing();
    }

    CloseWindow();
    return 0;
}