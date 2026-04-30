#include <iostream>
#include <string>

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "quit" || line == "exit") {
      break;
    }
    std::cout << "{\"result\":" << line << "}" << std::endl;
  }
  return 0;
}

