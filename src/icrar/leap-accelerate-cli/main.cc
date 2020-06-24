#include <iostream>
#include <string>
#include <CLI/CLI.hpp>

int main(int argc, char** argv)
{
  CLI::App app { " App description " };
  std::string filename = "default";
  app.add_option("-f,--file", filename, "A help string");
  try {
      app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
      return app.exit(e);
  }

  std::cout << " Hello World!" << std::endl;
}
