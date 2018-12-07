#ifndef BLOBMATH_H
#define BLOBMATH_H

#include "blob.h"
#include <vector>

std::vector<ScaleBlob*> longest_path(ScaleBlob* sb);
std::vector<std::vector<ScaleBlob*>> longest_paths(std::vector<ScaleBlob*> sb);

#endif