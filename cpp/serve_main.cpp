#include <iostream>
#include <string>

namespace {

std::string json_string_value(const std::string& line, const std::string& key) {
  const std::string quoted_key = "\"" + key + "\"";
  const auto key_pos = line.find(quoted_key);
  if (key_pos == std::string::npos) {
    return "";
  }
  const auto colon = line.find(':', key_pos + quoted_key.size());
  if (colon == std::string::npos) {
    return "";
  }
  const auto start = line.find('"', colon + 1);
  if (start == std::string::npos) {
    return "";
  }
  const auto end = line.find('"', start + 1);
  if (end == std::string::npos) {
    return "";
  }
  return line.substr(start + 1, end - start - 1);
}

std::string json_payload(const std::string& line) {
  const std::string key = "\"payload\"";
  const auto key_pos = line.find(key);
  if (key_pos == std::string::npos) {
    return "null";
  }
  const auto colon = line.find(':', key_pos + key.size());
  if (colon == std::string::npos) {
    return "null";
  }
  auto start = colon + 1;
  while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
    ++start;
  }
  auto end = line.size();
  while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t' || line[end - 1] == '}')) {
    --end;
  }
  return line.substr(start, end - start);
}

}  // namespace

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    const std::string op = json_string_value(line, "op");
    if (line == "quit" || line == "exit" || op == "exit") {
      break;
    }
    if (op == "health") {
      std::cout << "{\"status\":\"ok\"}" << std::endl;
    } else if (op == "predict") {
      std::cout << "{\"result\":" << json_payload(line) << "}" << std::endl;
    } else {
      std::cout << "{\"error\":\"unsupported op\"}" << std::endl;
    }
  }
  return 0;
}
