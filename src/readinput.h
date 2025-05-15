#pragma once
#include <string>
#include <vector>
#include <fstream> //for file
#include <iostream> //cout

#define COMMENT_CHAR '#'//注释符
#define CHK false
//using namespace std;
 
/****************************************************************************
 * read the input file : input_nn and input_density
 * comment begins with #
 * default format of content is "var=value"
 * *****************************************************************************/
class readinput{
//private:
public:
  std::ifstream *file_input;
  std::vector<std::string> content_in;
  //int id_line;
  size_t id_line;
public :
  readinput(const std::string & filename);

  bool IsSpecial(char c);
  void Trim(std::string & str);
  void                     get_content(const std::string & filename);
  std::string              get_string();
  int                      get_string2int();
  bool                     get_string2bool();
  double                   get_string2double();
  std::vector<std::string> get_string2string_arr();
  std::vector<int>         get_string2int_arr();
  std::vector<double>      get_string2double_arr();
  //std::vector<std::string> get_string_arr(std::string value_str);

};
