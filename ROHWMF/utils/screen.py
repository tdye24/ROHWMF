# ！usr/bin/python
# -*- coding:utf-8 -*-#
# @date:2019/1/22 14:30
# @name:screen
# @author:TDYe


def count_bracket(expression):
	left_bracket_count = 0
	right_bracket_count = 0
	for item in list(expression):
		if item == "(":
			left_bracket_count += 1
		elif item == ")":
			right_bracket_count += 1
	return [left_bracket_count, right_bracket_count]


def screen(expression):
	str_list = list(expression)
	length = str_list.__len__()
	for i in range(length):
		if count_bracket(expression)[0] != count_bracket(expression)[1]:
			if str_list[i] == "(" and str_list[i+1] == "(":
				str_list[i+1] = "1"
			elif str_list[i] == "(" and str_list[i+1] in ["+", "-", "*", "/"]:
				str_list[i] = "1"
			elif str_list[i] == "(" and i == length-1:
				str_list[i] == "1"
			elif str_list[i] == ")" and str_list[i+1] == ")":
				str_list[i] == "7"
			elif str_list[i] == ")" and str_list[i-1] in ["+", "-", "*", "/"]:
				str_list[i] == "7"
			elif str_list[i] == "+" and str_list[i+1] == "+":
				str_list[i+1] = "4"
			elif str_list[i] == "+" and (str_list[i+1] in ["-", "*", "/"] or str_list[i-1] in ["-", "*", "/"]):
				str_list[i] = "4"
			elif str_list[i] == "*" and (str_list[i+1] in ["+", "-", "*", "/"] or str_list[i-1] in ["+", "-", "*", "/"]):
				str_list[i] = "2"
			else:
				return "0+0"

	return "".join(str_list)
