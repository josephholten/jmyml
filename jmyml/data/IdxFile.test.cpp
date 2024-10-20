#include <gtest/gtest.h>
#include <jmyml/data/IdxFile.hpp>

TEST(IdxFile, Dimensions) {
  jmyml::IdxFile test_images;
  test_images.load(DATA_PATH "/test-images.idx");
  std::vector<int> images_sizes = {10000, 28, 28};
  EXPECT_EQ(test_images.shape, images_sizes);

  jmyml::IdxFile test_labels;
  test_labels.load(DATA_PATH "/test-labels.idx");
  std::vector<int> labels_sizes = {10000};
  EXPECT_EQ(test_labels.shape, labels_sizes);
}
