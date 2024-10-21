#ifndef JMYML_LOADIDX_HPP
#define JMYML_LOADIDX_HPP

#include <string>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cassert>
#include <fmt/core.h>

namespace jmyml {

struct IdxFile {
    enum class DataType {
        u8,
        i8,
        i16,
        i32,
        f32,
        f64,
        Null,
    };

    void* data = nullptr;
    std::vector<int> shape;
    DataType type = DataType::Null;

    // https://github.com/cvdfoundation/mnist?tab=readme-ov-file
    void load(const char* path) {
        std::fstream fs{path};

        // first two bytes must be zero
        char c = fs.get();
        if (c != 0) {
            fmt::println("magic number: not an idx file!");
            exit(1);
        }
        c = fs.get();
        if (c != 0) {
            fmt::println("magic number: not an idx file!");
            exit(1);
        }

        // next byte encodes type of data
        char t = fs.get();
        switch (t) {
        case 0x08:
            type = DataType::u8;
            break;

        case 0x09:
            type = DataType::i8;
            break;

        case 0x0B:
            type = DataType::i16;
            break;

        case 0x0C:
            type = DataType::i32;
            break;

        case 0x0D:
            type = DataType::f32;
            break;

        case 0x0E:
            type = DataType::f64;
            break;

        default:
            fmt::println("magic number: third byte '{:x}' not a valid idx file!", t);
            exit(1);
            break;
        }

        // next byte is number of dimensions
        uint8_t dimensions = fs.get();

        // next are the sizes of the dimensions, last is fastest
        size_t total_size = 1;
        for (size_t d = 0; d < dimensions; d++) {
            // the sizes are msb integers
            int size = 0;
            for (size_t i = 0; i < 4; i++)
                size |= fs.get() << (3 - i) * 8;
            shape.push_back(size);
            total_size *= size;
        }

        // next is the data itself
        if (data == nullptr)
            data = malloc(total_size);

        switch (type) {
            case DataType::u8:
                for (size_t i = 0; i < total_size; i++)
                    ((uint8_t*)data)[i] = fs.get();
                break;
            case DataType::i8:
                fmt::println("idx: i8 datatype '{:x}' unsupported", t);
                exit(1);
                break;
            case DataType::i16:
                fmt::println("idx: i16 datatype '{:x}' unsupported", t);
                exit(1);
                break;
            case DataType::i32:
                fmt::println("idx: i32 datatype '{:x}' unsupported", t);
                exit(1);
                break;
            case DataType::f32:
                fmt::println("idx: f32 datatype '{:x}' unsupported", t);
                exit(1);
                break;
            case DataType::f64:
                fmt::println("idx: f64 datatype '{:x}' unsupported", t);
                exit(1);
                break;
            case DataType::Null:
                fmt::println("idx: null datatype should not be deserialized", t);
                exit(1);
                break;
            default:
                fmt::println("idx: should not happen", t);
                exit(1);
                break;
        }
    };

    ~IdxFile() {
        free(data);
    }
};
};

#endif /* JMYML_LOADIDX_HPP */