#include <assert.h>
#include <iostream>

#include "reef/util/json.h"

namespace reef {
namespace util {

#define IS_DIGIT(chara) ((chara) <= '9' && (chara) >= '0')
#define IS_SPACE(chara) ((chara) == ' ' || (chara) == '\n' || (chara) == '\t')

#define MATCH_CONDITION(iter, str, delim) \
    ((iter) < (str).length() && ((str)[iter] != delim || (str)[iter-1] == '\\'))

#define TOKENIZE_SEPARATEOR(iter, type, sep, tokens) \
    case sep: \
        tokens.push_back(JsonParser::token(type)); \
        continue; 

#define TOKENIZE_COUPLED(iter, str, type, sep, tokens) \
    case sep: \
    { \
        size_t _tmp = iter + 1; \
        for (; MATCH_CONDITION(_tmp, str, sep); _tmp++) ; \
        tokens.push_back(JsonParser::token(type, str.substr(iter+1, _tmp-iter-1))); \
        iter = _tmp; \
        continue; \
    }

const char* token_name[] = {"invalid", "string", "number", "[", "]", "{", "}", ",", ":"};


JsonObject* JsonParser::parse(std::string& str) {
    int top = 0;
    JsonObject* jobj = _parse(tokenize(str), top);
    return jobj;
}

std::vector<JsonParser::token> JsonParser::tokenize(std::string& str) {
    std::vector<JsonParser::token> tokens;

    while (!str.empty()) {
        strip_space(str);
        std::string token = split_by_space(str);

        for (size_t i = 0; i < token.length(); i++) {
            switch (token[i]) {
                TOKENIZE_COUPLED(i, token, STRING, '"', tokens);
                TOKENIZE_COUPLED(i, token, STRING, '\'', tokens);
                TOKENIZE_SEPARATEOR(i, COMMA, ',', tokens);
                TOKENIZE_SEPARATEOR(i, LBRACKET, '[', tokens);
                TOKENIZE_SEPARATEOR(i, RBRACKET, ']', tokens);
                TOKENIZE_SEPARATEOR(i, LBRACE, '{', tokens);
                TOKENIZE_SEPARATEOR(i, RBRACE, '}', tokens);
                TOKENIZE_SEPARATEOR(i, COLON, ':', tokens);
            }

            if (token[i] == '-' || IS_DIGIT(token[i])) {
                size_t tmp = (token[i] == '-') ? (i + 1) : i;
                bool is_float = false;
                for (; tmp < token.length() && (IS_DIGIT(token[tmp]) || token[tmp] == '.'); tmp++) 
                    if (token[tmp] == '.') is_float = true;
                tokens.push_back(JsonParser::token(is_float ? FLOAT : INTEGER, token.substr(i, tmp-i)));
                i = tmp - 1;
                continue;
            }

            printf("Error: unrecognizable token at %s\n", token.substr(i).c_str());
            exit(1);
        }
    }

    return tokens;
}

JsonObject* JsonParser::_parse(std::vector<token> tokens, int& top) {
    JsonObject* cur = new JsonObject;

    switch (tokens[top].type) {
    case LBRACE:
        cur->type = JsonObject::J_DICT;
        top++;
        while (tokens[top].type != RBRACE) {
            assert(tokens[top].type == STRING);
            std::string key = tokens[top].value;

            assert(tokens[top+1].type == COLON);
            top += 2;

            cur->mval.insert(std::pair<std::string, JsonObject*>(key, _parse(tokens, top)));
            if (tokens[top].type == COMMA) top++;
        }
        top++;
        return cur;

    case LBRACKET:
        cur->type = JsonObject::J_LIST;
        top++;
        while (tokens[top].type != RBRACKET) {
            cur->lval.push_back(_parse(tokens, top));
            if (tokens[top].type == COMMA) top++;
        }
        top++;
        return cur;
    
    case INTEGER:
        cur->type = JsonObject::J_INT;
        cur->ival = atoi(tokens[top].value.c_str());
        top++;
        return cur;
    case FLOAT:
        cur->type = JsonObject::J_FLOAT;
        cur->fval = (float)atof(tokens[top].value.c_str());
        top++;
        return cur;
    case STRING:
        cur->type = JsonObject::J_STRING;
        cur->sval = tokens[top].value;
        top++;
        return cur;
    default:
        break;
    }

    return cur;
}

void JsonParser::strip_space(std::string& str) {

    for (size_t i = 0; i < str.length(); i++) {
        if (!IS_SPACE(str[i])) {
            str.erase(0, i);
            return;
        }
    }
}

std::string JsonParser::split_by_space(std::string& str) {
    std::string token;
    
    for (size_t i = 0; i < str.length(); i++) {
        switch (str[i]) {
            case '"':
                for (i++; MATCH_CONDITION(i, str, '"'); i++) ;
                break;
            case '\'':
                for (i++; MATCH_CONDITION(i, str, '\''); i++) ;
                break;
            case ' ':
            case '\n':
            case '\t':
                token = str.substr(0, i);
                str.erase(0, i);
                return token;
            default:
                continue;
        }
    }

    token = str;
    str.erase(0, str.length());
    return token;
}
} // namespace util
} // namespace reef

