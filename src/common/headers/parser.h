#ifndef PARSER_H
#define PARSER_H

#include "context.h"

class Parser
{
  public:
    Parser() {}

    void parse_input_file(char *input_file, Context& context);
};

#endif // PARSER_H
