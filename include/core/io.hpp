#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace fpstudy::core {

class CsvWriter {
public:
    explicit CsvWriter(const std::filesystem::path& path, bool append = false);
    void write_header(const std::vector<std::string>& columns);
    void write_row(const std::vector<std::string>& values);

private:
    std::ofstream stream_;
    bool header_written_ = false;
};

namespace json {

struct Value;

using Object = std::unordered_map<std::string, Value>;
using Array = std::vector<Value>;

struct Value {
    using Variant = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;
    Variant data;

    Value() : data(std::nullptr_t{}) {}
    Value(std::nullptr_t) : data(std::nullptr_t{}) {}
    Value(bool v) : data(v) {}
    Value(double v) : data(v) {}
    Value(int v) : data(static_cast<double>(v)) {}
    Value(std::string s) : data(std::move(s)) {}
    Value(const char* s) : data(std::string(s)) {}
    Value(Array arr) : data(std::move(arr)) {}
    Value(Object obj) : data(std::move(obj)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }
    bool is_number() const { return std::holds_alternative<double>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<Array>(data); }
    bool is_object() const { return std::holds_alternative<Object>(data); }

    const Array& as_array() const { return std::get<Array>(data); }
    Array& as_array() { return std::get<Array>(data); }

    const Object& as_object() const { return std::get<Object>(data); }
    Object& as_object() { return std::get<Object>(data); }

    const std::string& as_string() const { return std::get<std::string>(data); }
    double as_number() const { return std::get<double>(data); }
    bool as_bool() const { return std::get<bool>(data); }
};

Value parse(const std::string& text);
Value load_file(const std::filesystem::path& path);

std::string serialize_compact(const Value& value);

} // namespace json

} // namespace fpstudy::core

