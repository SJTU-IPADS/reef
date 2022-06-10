#pragma once

#include <string>
#include <vector>
#include <map>

namespace reef {
namespace util {

// A json object, which supports `string`, `int`, `float`, `list` and `dict(map)`.
class JsonObject {
public:
    std::string sval;
    uint32_t ival;
    float fval;
    std::vector<JsonObject*> lval;
    std::map<std::string, JsonObject*> mval;

    enum jobject_type {J_STRING, J_INT, J_FLOAT, J_LIST, J_DICT};
    jobject_type type;

    JsonObject() {}
};

// A json parser that parses a string to a JsonObject.
class JsonParser {
public:
    // parses a string to a JsonObject
    static JsonObject* parse(std::string& str);

private:
    enum token_type {INVAL, STRING, FLOAT, INTEGER, LBRACKET, RBRACKET, LBRACE, RBRACE, COMMA, COLON};
    struct token {
        token_type type;
        std::string value;
        token(token_type t, std::string v="") : type(t), value(v) {}
    };

    static std::vector<token> tokenize(std::string& str);
    static JsonObject* _parse(std::vector<token> tokens, int& top);

    static void strip_space(std::string& str);
    static std::string split_by_space(std::string& str);
};

} // namespace reef
} // namespace util