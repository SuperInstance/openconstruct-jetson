#include "openconstruct-jetson.hpp"
#include <algorithm>
#include <regex>
#include <sstream>

namespace openconstruct {
namespace jetson {

class PlatoJetson {
public:
  PlatoJetson(OpenConstructJetson *parent) : parent_(parent) {}

  /**
   * @brief Process a command from the Plato network
   */
  std::string process(const std::string &raw_cmd) {
    std::string cmd = trim(raw_cmd);
    if (cmd.empty()) {
      return "ERROR: Empty command";
    }

    // Parse command and arguments
    std::vector<std::string> tokens = split(cmd, ' ');
    std::string command = tokens[0];
    std::vector<std::string> args(tokens.begin() + 1, tokens.end());

    // Dispatch command
    if (command == "status") {
      return handle_status(args);
    } else if (command == "describe") {
      return handle_describe(args);
    } else if (command == "restart") {
      return handle_restart(args);
    } else if (command == "camera") {
      return handle_camera(args);
    } else if (command == "microphone") {
      return handle_microphone(args);
    } else if (command == "help") {
      return handle_help();
    } else if (command == "echo") {
      return handle_echo(args);
    } else if (command == "ping") {
      return "PONG";
    } else {
      return "ERROR: Unknown command '" + command +
             "'. Type 'help' for available commands.";
    }
  }

  std::string get_prompt() const { return "plato-jetson> "; }

private:
  OpenConstructJetson *parent_;

  std::string handle_status(const std::vector<std::string> & /*args*/) {
    return parent_->system_status();
  }

  std::string handle_describe(const std::vector<std::string> &args) {
    if (args.empty()) {
      return parent_->describe_scene() + "\n" + parent_->describe_audio();
    } else if (args[0] == "scene") {
      return parent_->describe_scene();
    } else if (args[0] == "audio") {
      return parent_->describe_audio();
    } else {
      return "ERROR: Unknown describe target. Use 'scene' or 'audio'.";
    }
  }

  std::string handle_restart(const std::vector<std::string> & /*args*/) {
    // In a real implementation, this would restart the service
    // For now, return success
    return "RESTART: Restart command received. Service restart initiated.";
  }

  std::string handle_camera(const std::vector<std::string> &args) {
    if (args.empty()) {
      return "ERROR: camera command requires arguments. Use 'camera list', "
             "'camera add <id> <name>', or 'camera remove <id>'.";
    }

    if (args[0] == "list") {
      std::ostringstream oss;
      oss << "Registered cameras:\n";
      // Access parent's camera map (would need friend or accessor)
      oss << "  (List not available in this interface)\n";
      return oss.str();
    } else if (args[0] == "add" && args.size() >= 3) {
      int device_id;
      try {
        device_id = std::stoi(args[1]);
      } catch (const std::exception &) {
        return "ERROR: Camera device ID must be an integer, got '" + args[1] +
               "'";
      }
      std::string name = args[2];
      parent_->register_camera(device_id, name.c_str());
      return "OK: Camera '" + name + "' registered with ID " +
             std::to_string(device_id);
    } else if (args[0] == "remove" && args.size() >= 2) {
      return "OK: Camera removal requested (not implemented)";
    } else {
      return "ERROR: Invalid camera command";
    }
  }

  std::string handle_microphone(const std::vector<std::string> &args) {
    if (args.empty()) {
      return "ERROR: microphone command requires arguments. Use 'microphone "
             "list', 'microphone add <id> <name>', or 'microphone remove "
             "<id>'.";
    }

    if (args[0] == "list") {
      std::ostringstream oss;
      oss << "Registered microphones:\n";
      oss << "  (List not available in this interface)\n";
      return oss.str();
    } else if (args[0] == "add" && args.size() >= 3) {
      int device_id;
      try {
        device_id = std::stoi(args[1]);
      } catch (const std::exception &) {
        return "ERROR: Microphone device ID must be an integer, got '" +
               args[1] + "'";
      }
      std::string name = args[2];
      parent_->register_microphone(device_id, name.c_str());
      return "OK: Microphone '" + name + "' registered with ID " +
             std::to_string(device_id);
    } else if (args[0] == "remove" && args.size() >= 2) {
      return "OK: Microphone removal requested (not implemented)";
    } else {
      return "ERROR: Invalid microphone command";
    }
  }

  std::string handle_help() {
    std::ostringstream oss;
    oss << "Available commands:\n";
    oss << "  status              - Show system status (GPU, CPU, memory, "
           "thermal)\n";
    oss << "  describe [scene|audio] - Describe current scene or audio\n";
    oss << "  restart             - Restart the OpenConstruct service\n";
    oss << "  camera list         - List registered cameras\n";
    oss << "  camera add <id> <name>  - Register a camera\n";
    oss << "  microphone list     - List registered microphones\n";
    oss << "  microphone add <id> <name> - Register a microphone\n";
    oss << "  ping                - Test connection (returns PONG)\n";
    oss << "  echo <text>         - Echo text back\n";
    oss << "  help                - Show this help message\n";
    return oss.str();
  }

  std::string handle_echo(const std::vector<std::string> &args) {
    std::ostringstream oss;
    for (size_t i = 0; i < args.size(); ++i) {
      if (i > 0)
        oss << " ";
      oss << args[i];
    }
    return oss.str();
  }

  std::string trim(const std::string &s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos)
      return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
  }

  std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
      if (!token.empty()) {
        tokens.push_back(token);
      }
    }
    return tokens;
  }
};

// Free function declared in the public header so
// OpenConstructJetson::process_command (in jetson_status.cpp) can delegate to
// the shell without seeing the PlatoJetson class definition, which is private
// to this translation unit.
std::string process_plato_command(OpenConstructJetson *node,
                                  const std::string &raw_cmd) {
  PlatoJetson shell(node);
  return shell.process(raw_cmd);
}

} // namespace jetson
} // namespace openconstruct