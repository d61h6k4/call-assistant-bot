
namespace aikit::ml {
// A bounding box. The box is defined by its upper left corner (xmin, ymin)
// and its width and height, all in coordinates normalized by the image
// dimensions.
struct Detection {
  float xmin;
  float ymin;
  float width;
  float height;
  int label_id;
  float score;
};
} // namespace aikit::ml
