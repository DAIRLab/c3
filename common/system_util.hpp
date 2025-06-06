// Standard library includes.
#include <fstream>
#include <regex>
#include <string>

#include <drake/systems/framework/system_visitor.h>

namespace c3 {
namespace common {
// Utility function to save and visualize the system diagram.
void DrawAndSaveDiagramGraph(const drake::systems::Diagram<double>& diagram,
                             std::string path) {
  if (path.empty()) path = "../" + diagram.get_name();  // Default path.

  // Save Graphviz string to a file.
  std::ofstream out(path);
  out << diagram.GetGraphvizString();
  out.close();

  // Convert Graphviz string to an image file.
  std::regex r(" ");
  path = std::regex_replace(path, r, "\\ ");
  std::string cmd = "dot -Tps " + path + " -o " + path + ".ps";
  (void)std::system(cmd.c_str());

  // Remove the temporary Graphviz string file.
  cmd = "rm " + path;
  (void)std::system(cmd.c_str());
}
}  // namespace common
}  // namespace c3