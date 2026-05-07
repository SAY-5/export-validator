// Minimal stable JSON writer aligned with the Python compare report format.
#pragma once

#include <string>

#include "export_validator/types.h"

namespace export_validator {

std::string serialize(const Report& report);

}  // namespace export_validator
