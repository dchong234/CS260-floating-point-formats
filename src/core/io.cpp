#include "core/io.hpp"

#include <charconv>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace fpstudy::core {

namespace {

std::string escape_csv_field(std::string_view value) {
    bool needs_quotes = value.find_first_of(",\"\n\r") != std::string_view::npos;
    if (!needs_quotes) {
        return std::string(value);
    }
    std::string quoted;
    quoted.reserve(value.size() + 2);
    quoted.push_back('"');
    for (char c : value) {
        if (c == '"') {
            quoted.push_back('"');
        }
        quoted.push_back(c);
    }
    quoted.push_back('"');
    return quoted;
}

} // namespace

CsvWriter::CsvWriter(const std::filesystem::path& path, bool append) {
    if (!path.has_parent_path()) {
        std::filesystem::create_directories(".");
    } else {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ios_base::openmode mode = std::ios::out;
    if (append) {
        mode |= std::ios::app;
    } else {
        mode |= std::ios::trunc;
    }
    stream_.open(path, mode);
    if (!stream_) {
        throw std::runtime_error("Failed to open CSV file: " + path.string());
    }
    if (append && std::filesystem::exists(path) && std::filesystem::file_size(path) > 0) {
        header_written_ = true;
    }
}

void CsvWriter::write_header(const std::vector<std::string>& columns) {
    if (header_written_) {
        return;
    }
    std::string line;
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) line += ",";
        line += escape_csv_field(columns[i]);
    }
    stream_ << line << "\n";
    header_written_ = true;
}

void CsvWriter::write_row(const std::vector<std::string>& values) {
    if (!header_written_) {
        throw std::runtime_error("CSV header must be written before rows");
    }
    std::string line;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) line += ",";
        line += escape_csv_field(values[i]);
    }
    stream_ << line << "\n";
}

namespace json {

namespace {

class Parser {
public:
    explicit Parser(const std::string& input) : text_(input) {}

    Value parse_value() {
        skip_ws();
        if (match("null")) return Value{std::nullptr_t{}};
        if (match("true")) return Value{true};
        if (match("false")) return Value{false};
        if (peek() == '"') return Value{parse_string()};
        if (peek() == '{') return Value{parse_object()};
        if (peek() == '[') return Value{parse_array()};
        return Value{parse_number()};
    }

private:
    const std::string& text_;
    size_t pos_ = 0;

    char peek() {
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Unexpected end of JSON input");
        }
        return text_[pos_];
    }

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    bool match(std::string_view token) {
        skip_ws();
        if (text_.substr(pos_, token.size()) == token) {
            pos_ += token.size();
            return true;
        }
        return false;
    }

    std::string parse_string() {
        skip_ws();
        if (peek() != '"') throw std::runtime_error("Expected string");
        ++pos_;
        std::string result;
        while (pos_ < text_.size()) {
            char c = text_[pos_++];
            if (c == '"') break;
            if (c == '\\') {
                if (pos_ >= text_.size()) throw std::runtime_error("Invalid escape");
                char esc = text_[pos_++];
                switch (esc) {
                    case '"': result.push_back('"'); break;
                    case '\\': result.push_back('\\'); break;
                    case '/': result.push_back('/'); break;
                    case 'b': result.push_back('\b'); break;
                    case 'f': result.push_back('\f'); break;
                    case 'n': result.push_back('\n'); break;
                    case 'r': result.push_back('\r'); break;
                    case 't': result.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) throw std::runtime_error("Invalid unicode escape");
                        unsigned int code = 0;
                        for (int i = 0; i < 4; ++i) {
                            char hex = text_[pos_++];
                            code <<= 4;
                            if (hex >= '0' && hex <= '9') code |= hex - '0';
                            else if (hex >= 'a' && hex <= 'f') code |= 10 + hex - 'a';
                            else if (hex >= 'A' && hex <= 'F') code |= 10 + hex - 'A';
                            else throw std::runtime_error("Invalid unicode hex escape");
                        }
                        if (code <= 0x7F) {
                            result.push_back(static_cast<char>(code));
                        } else {
                            throw std::runtime_error("Unicode escapes beyond ASCII unsupported");
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error("Invalid escape character");
                }
            } else {
                result.push_back(c);
            }
        }
        return result;
    }

    double parse_number() {
        skip_ws();
        size_t start = pos_;
        if (text_[pos_] == '-') ++pos_;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (text_[pos_] == '+' || text_[pos_] == '-') ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        auto len = pos_ - start;
        std::string token = text_.substr(start, len);
        return std::stod(token);
    }

    Object parse_object() {
        expect('{');
        skip_ws();
        Object obj;
        if (peek() == '}') {
            ++pos_;
            return obj;
        }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            expect(':');
            Value value = parse_value();
            obj.emplace(std::move(key), std::move(value));
            skip_ws();
            char c = peek();
            ++pos_;
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("Expected comma in object");
        }
        return obj;
    }

    Array parse_array() {
        expect('[');
        skip_ws();
        Array arr;
        if (peek() == ']') {
            ++pos_;
            return arr;
        }
        while (true) {
            Value value = parse_value();
            arr.push_back(std::move(value));
            skip_ws();
            char c = peek();
            ++pos_;
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("Expected comma in array");
        }
        return arr;
    }

    void expect(char c) {
        skip_ws();
        if (peek() != c) {
            throw std::runtime_error("Expected character");
        }
        ++pos_;
    }
};

std::string serialize_simple(const Value& value);

std::string serialize_simple(const Value& value) {
    struct Visitor {
        std::string operator()(std::nullptr_t) const { return "null"; }
        std::string operator()(bool v) const { return v ? "true" : "false"; }
        std::string operator()(double v) const {
            std::ostringstream oss;
            oss << v;
            return oss.str();
        }
        std::string operator()(const std::string& s) const {
            std::string out = "\"";
            for (char c : s) {
                if (c == '"' || c == '\\') out.push_back('\\');
                out.push_back(c);
            }
            out.push_back('"');
            return out;
        }
        std::string operator()(const Array& arr) const {
            std::string out = "[";
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) out += ",";
                out += serialize_simple(arr[i]);
            }
            out += "]";
            return out;
        }
        std::string operator()(const Object& obj) const {
            std::string out = "{";
            size_t idx = 0;
            for (const auto& [key, val] : obj) {
                if (idx++ > 0) out += ",";
                out += (*this)(key);
                out += ":";
                out += serialize_simple(val);
            }
            out += "}";
            return out;
        }
        std::string operator()(const Value& value) const {
            return std::visit(*this, value.data);
        }
    };
    return Visitor{}(value);
}

} // namespace

Value parse(const std::string& text) {
    Parser parser(text);
    return parser.parse_value();
}

Value load_file(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open JSON file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << ifs.rdbuf();
    return parse(buffer.str());
}

std::string serialize_compact(const Value& value) {
    return serialize_simple(value);
}

} // namespace json

} // namespace fpstudy::core

