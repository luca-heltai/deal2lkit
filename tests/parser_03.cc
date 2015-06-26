//-----------------------------------------------------------
//
//    Copyright (C) 2014 by the deal.II authors
//
//    This file is subject to LGPL and may not be distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//-----------------------------------------------------------

#include "tests.h"

// Try parsing enumerator


#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <iostream>
#include <string>

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_lit.hpp>

#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include <string>
#include <vector>
#include <deal.II/base/utilities.h>

// Parse a LinearOperator by name, taken from a global map.

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace phoenix = boost::phoenix;

typedef Vector<double> VEC;

int counter = 0;

LinearOperator<VEC> named_op(std::string name)
{
  LinearOperator<VEC> op;
  op.vmult = [name](VEC &, const VEC &)
  {
    deallog << "v" << (counter+1) << " = [ " << name << " * v"
            << counter << " ]" << std::endl;
    ++counter;
  };

  op.Tvmult = [name](VEC &, const VEC &)
  {
    deallog << "v" << (counter+1) << " = [ T(" << name << ") * v"
            << counter << " ]" << std::endl;
    ++counter;
  };
  op.vmult_add = [name](VEC &, const VEC &)
  {
    deallog << "v" << (counter+1) << " += [ " << name << " * v"
            << counter << " ]" << std::endl;
    ++counter;
  };

  op.Tvmult_add = [name](VEC &, const VEC &)
  {
    deallog << "v" << (counter+1) << " += [ T(" << name << ") * v"
            << counter << " ]" << std::endl;
    ++counter;
  };
  op.reinit_domain_vector = [](VEC &, bool)
  {
    return;
  };
  op.reinit_range_vector = [](VEC &, bool)
  {
    return;
  };
  return op;
};

std::map<std::string, LinearOperator<VEC> > op_map;

int main()
{
  initlog();
  namespace qi = boost::spirit::qi;
  namespace ascii = boost::spirit::ascii;
  namespace phoenix = boost::phoenix;

  using qi::double_;
  using qi::_1;
  using ascii::space;
  using phoenix::ref;

  using qi::lit;
  using qi::double_;
  using qi::int_;
  using qi::phrase_parse;
  using ascii::space;


  using qi::_1;
  using phoenix::ref;


  for ( auto &s : Utilities::split_string_list("A,B,C") )
    op_map[s] = named_op(s);



  struct LinearOperator_ : qi::symbols<char, LinearOperator<VEC> >
  {
    LinearOperator_()
    {
      for (auto &op : op_map)
        add(op.first, op.second);
    };
  } lin_op_;




  VEC a;

  LinearOperator<VEC> op;


  // A row of the matrix: comma separated list of strings
  auto grammar =
    lin_op_[ref(op) = _1] >> *('*' >> lin_op_[ref(op) *= _1]);


  std::vector<std::string> str_list;
  str_list.push_back("A*B*C");

  for (unsigned int i=0; i<str_list.size(); ++i)
    {
      std::string &str = str_list[i];


      bool r = phrase_parse(str.begin(), str.end(), grammar, space);

      if (r)
        {
          deallog << "-------------------------" <<std::endl;
          deallog << "Parsing succeeded: " << str << std::endl;
          op.vmult(a, a);
          deallog << "-------------------------" <<std::endl;
        }
      else
        {
          deallog << "-------------------------" << std::endl;
          deallog << "Parsing failed for " << str << std::endl;
          deallog << "-------------------------" << std::endl;
        }
    }

  return 0;
}

