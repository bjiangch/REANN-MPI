#include "readinput.h"
using namespace std;

bool readinput::IsSpecial(char c){
  if (c==' '||c=='\t'||c=='\"'||c=='\'')
    return true;
  return false;
}

//strip whitespace char to get compact non-empty string
void readinput::Trim(std::string & str) {
  if (str.empty()) return;
  //int i, pos_start, pos_end;
  size_t i;
  for (i = 0; i < str.size(); ++i) {
    if(str[i]=='#') break;
  }
  int id_eff=i;

  if(id_eff==0){
    str="";
    return;
  }else {
    str=str.substr(0,id_eff);
  }

  std::string str_new="";
  for (i = 0; i < str.size(); ++i) {
    if (!IsSpecial(str[i])) {
      str_new+=str[i];
    }
  }
  str=str_new;
}

readinput::readinput(const string & filename){
  //initilize id_line;
  id_line=0;
  file_input=new ifstream(filename.c_str());
  //if (!file_input) {
  //  cout << "ERROR! Could find the file " << filename << endl;
  //  std::exit(1);
  //}
  std::string line;
  while (getline(*file_input, line)) {
    Trim(line);
    int pos_now = line.find('=');
    if (pos_now>0) {
      if (!line.empty()) content_in.push_back(line);
    }
  }
  #if CHK
    std::cout<<std::string(80, '=')<<std::endl;
    std::cout<<"Reading "<< filename << " , ";
    std::cout<< "Total lines = "<<content_in.size()<<std::endl;
    int i_line=0;
    char i_line_str[8]={0};
    for (auto iter : content_in) {
      sprintf(i_line_str, "%02d", i_line++);
      std::cout<<"line " <<i_line_str<< " : "<< iter << std::endl;
    }
  #endif
}


std::string readinput::get_string() {

  if (id_line>=content_in.size()){
    std::cout<<"\n\nERROR! OUT OF RANGE OF FILE\n"<<std::endl;
    std::cout<<"CHECK THE EXISTENCE AND CONTENT OF THE FILE\n\n"<<std::endl;
    std::exit(1);
  }

  std::string line=content_in[id_line];
  id_line+=1;

  int pos_now = line.find('=');
  int pos_end = line.size()-1;

  string var_name=line.substr(0, pos_now);
  Trim(var_name);
  string value_str=line.substr(pos_now+1, pos_end+1-(pos_now+1));
  Trim(value_str);

  #if CHK
    std::cout<<std::string(80, '-')<<std::endl;
    std::cout<<"var name  str "<< var_name  << std::endl;
    std::cout<<"var value str "<< value_str << std::endl;
  #endif

  return value_str;
}

int readinput::get_string2int() {
  std::string value_str=get_string();
  #if CHK
    std::cout<<"var value int "<< stoi(value_str) << std::endl;
  #endif
  return stoi(value_str);
}

bool readinput::get_string2bool() {
  std::string value_str=get_string();
  bool value_bool;
  if (value_str=="True") {
    value_bool=true;
  }else{
    value_bool= false;
  }
  #if CHK
    std::cout<<"var value bool "<< value_bool << std::endl;
  #endif
  return value_bool;
}

double readinput::get_string2double() {
  std::string value_str=get_string();
  #if CHK
    std::cout<<"var value double "<< stod(value_str)  << std::endl;
  #endif
  return stod(value_str);
}

std::vector<std::string> readinput::get_string2string_arr() {
  std::string value_str=get_string();
  //the form of value should be "[...]"
  std::vector<string> val_str_arr;
  std::vector<int> id_comma={0};
  //for (int i = 1; i < value_str.size()-1; ++i) {
  for (size_t i = 1; i < value_str.size()-1; ++i) {
    if(value_str[i]==',') {
      id_comma.push_back(i);
    }
  }
  id_comma.push_back(value_str.size()-1);
  //for (int i = 0; i < id_comma.size()-1; ++i) {
  for (size_t i = 0; i < id_comma.size()-1; ++i) {
    std::string value_tmp=value_str.substr(id_comma[i]+1,id_comma[i+1]-id_comma[i]-1);
    val_str_arr.push_back(value_tmp);
  }
  #if CHK
    std::cout<<"var value str_arr " << std::endl;
    for (auto iter : val_str_arr) {
      std::cout << iter <<",";
    }
    std::cout << std::endl;
  #endif
  return val_str_arr;
}

std::vector<int> readinput::get_string2int_arr() {
  std::vector<std::string> arr_str=get_string2string_arr();
  std::vector<int> arr_int={};
  //for (int i = 0; i < arr_str.size(); ++i) {
  for (size_t i = 0; i < arr_str.size(); ++i) {
    arr_int.push_back(stoi(arr_str[i]));
  }
  #if CHK
    std::cout<<"var value arr_int " << std::endl;
    for (auto iter : arr_int) {
      std::cout << iter <<",";
    }
    std::cout << std::endl;
  #endif
  return arr_int;
}

std::vector<double> readinput::get_string2double_arr() {
  std::vector<std::string> arr_str=get_string2string_arr();
  std::vector<double> arr_double={};
  //for (int i = 0; i < arr_str.size(); ++i) {
  for (size_t i = 0; i < arr_str.size(); ++i) {
    arr_double.push_back(stod(arr_str[i]));
  }
  #if CHK
    std::cout<<"var value arr_double " << std::endl;
    for (auto iter : arr_double) {
      std::cout << iter <<",";
    }
    std::cout << std::endl;
  #endif
  return arr_double;
}
