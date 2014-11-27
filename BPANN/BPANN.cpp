// BPANN.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

const double ETA = 0.1;

vector<vector<vector<double>>> delta_w;

inline double Random(double low, double high)
{
	return low + (high-low)*rand()*1.0/RAND_MAX;
}
inline double Sigmoid(double x){
	return 1.0/(1.0+exp(-x));
}
void Init_neuron(vector<vector<double>> &o, int layer_num, const vector<int> &neu_num, double init)
{
	for(int i = 0; i < layer_num; ++i){
		vector<double> tmp;
		for(int j = 0; j < neu_num[i]; ++j)
			tmp.push_back(init);
		o.push_back(tmp);
	}
}
void Init_weights(vector<vector<vector<double>>> &w, int layer_num, const vector<int> &neu_num)
{
	int i, j, k;
	//随机初始化权值
	for (i = 0; i < layer_num-1; ++i) //第i层神经元
	{
		vector<vector<double>> tmp; 
		for (j = 0; j < neu_num[i]; ++j) //第i层的神经元个数
		{
			vector<double> tmp_w;
			for(k = 0; k < neu_num[i+1]; ++k)
			{
				double t = Random(0, 0.1);
				tmp_w.push_back(t);
			};
			tmp.push_back(tmp_w);
		}
		w.push_back(tmp);
		delta_w.push_back(tmp);
	}
}
double Cal_output(const vector<double> &x, const vector<double> &w, int n)
{
	double sum = 0.0;
	for(int i = 0; i < n; ++i){
		sum += x[i]*w[i];
	}
	return Sigmoid(sum);
}
void Cal_NN_outputs(const vector<vector<vector<double>>> &w, vector<vector<double>> &neu_output, const vector<double> &input, int layer_num, const vector<int> &neu_num)
{
	for (int i = 1; i < layer_num; ++i)
	{
		for (int j = 0; j < neu_num[i]; ++j){
			vector<double> tmp;
			for (int k = 0; k < neu_num[i-1]; ++k)
				tmp.push_back(w[i-1][k][j]);
			neu_output[i][j] = Cal_output(neu_output[i-1], tmp, neu_num[i-1]);
		}
	}
}
void Cal_output_error(vector<double> &e, const vector<double> &o, const vector<double> &t)
{
	int n = t.size();
	for(int k = 0; k < n; ++k)
		e[k] = o[k]*(1-o[k])*(t[k]-o[k]);
}
double Cal_hidden_error(const double out, const vector<double> &e, const vector<double> &w)
{
	int n = e.size();
	double sum = 0.0;
	for(int h = 0; h < n; ++h)
		sum += e[h]*w[h];
	return out*(1-out)*sum;
}
void Cal_errors(const vector<vector<vector<double>>> &w, vector<vector<double>> &errors, const vector<vector<double>> &neu_output, int layer_num, const vector<int> &neu_num, const vector<double> &output)
{
	Cal_output_error(errors[layer_num-1], neu_output[layer_num-1], output);//计算每个输出单元的误差
	for (int i = layer_num-2; i >= 0; --i) //计算隐藏层神经元的误差
	{
		for (int j = 0; j < neu_num[i]; ++j){
			
			errors[i][j] = Cal_hidden_error(neu_output[i][j], errors[i+1], w[i][j]);
		}
	}
}
double Cal_NNerror(const vector<double> &o, const vector<double> &output){
	double sum = 0.0;
	for(int i = 0; i < output.size(); ++i)
		sum += (o[i]-output[i])*(o[i]-output[i]);
	return sum;
}
void Update_weights(vector<vector<vector<double>>> &w, vector<vector<double>> &errors, const vector<vector<double>> &neu_output, int layer_num, const vector<int> &neu_num)
{
	int i, j, k;
	for(i = 0; i < layer_num-1; ++i){
		for(j = 0; j < neu_num[i]; ++j){
			for (k = 0; k < neu_num[i+1]; ++k){
				delta_w[i][j][k] = ETA*errors[i+1][k]*neu_output[i][j];
				w[i][j][k] += delta_w[i][j][k];
				//w[i][j][k] += ETA*errors[i][j]*neu_output[i][j];
			}
		}
	}
}
void print_output(const vector<double> &o, const vector<double> &t, const vector<double> &e)
{
	int n = t.size();
	for(int i = 0; i < n; ++i)
		cout << i << '\t' << t[i] << '\t' << o[i] << '\t' << e[i] << endl;
	cout << endl;
}

void print_outputs(ofstream &out, const vector<vector<double>> &o)
{
	for (int i = 0; i < o.size(); ++i)
	{
		for (int j = 0; j < o[i].size(); ++j)
			out << o[i][j] << '\t';
		out << endl;
	}
	out << endl;
}

void print_weights(ofstream &out, const vector<vector<vector<double>>> &w, const int layer_num, const vector<int> neu_num)
{
	for(int i = 0; i < layer_num-1; ++i)
	{
		out << "第" <<i+1<<"层"<<endl;
		for (int j = 0; j < neu_num[i]; ++j)
		{
			for (int k = 0; k < neu_num[i+1]; ++k)
				out << w[i][j][k] << '\t';
			out << endl;
		}
	}
	out << endl;
}
void NN(int layer_num, const vector<int> &neu_num, const vector<double> &input, const vector<double> &output)
{
	//weights[i][j][k]表示第i层的第j个神经元到第i+1层的第k个神经元的权值
	//errors[i][j]表示第i层的第j个神经元的误差
	//neu_output[i][j]表示第i层的第j个神经元的输出
	vector<vector<vector<double>>> weights;
	vector<vector<double>> errors;
	vector<vector<double>> neu_output;
	Init_weights(weights, layer_num, neu_num); //初始化权值
	Init_neuron(errors, layer_num, neu_num, 0.0); //初始化errors
	Init_neuron(neu_output, layer_num, neu_num, 0.0);//初始化每个神经元的输出
	for(int i = 0; i < input.size(); ++i)
		neu_output[0][i] = input[i];
	double e = 0.0;
	int count = 0;
	ofstream o_out("output.txt");
	ofstream w_out("weights.txt");
	ofstream e_out("errors.txt");
	ofstream delw_out("deltaw.txt");
	print_weights(w_out, weights, layer_num, neu_num);
	do{
		Cal_NN_outputs(weights, neu_output, input, layer_num, neu_num); //计算每个神经元的输出
		Cal_errors(weights, errors, neu_output, layer_num, neu_num, output); //计算每个神经元的误差
		e = Cal_NNerror(neu_output[layer_num-1], output);
		Update_weights(weights, errors, neu_output, layer_num, neu_num); //更新权值
		
		print_output(neu_output[layer_num-1], output, errors[layer_num-1]);
		cout << count++ << "\t: " << e << endl;
		o_out<<count<<endl;
		e_out<<count<<endl;
		delw_out<<count<<endl;
		w_out<<count<<endl;
		print_outputs(o_out, neu_output);
		print_outputs(e_out, errors);
		print_weights(delw_out, delta_w, layer_num, neu_num);
		print_weights(w_out, weights, layer_num, neu_num);
	}
	//while(count < 100);
	while(e>0.01);
}




int _tmain(int argc, _TCHAR* argv[])
{
	int input_num = 3;
	int output_num = 3;
	double input[] = {1.0, 0.25, -0.5};
	double output[] = {1.0, -1.0, 0.0};
	int layer_num = 3;
	int neu_num[] = {3, 2, 3};
	/*int input_num = 2;
	int output_num = 1;
	double input[] = {1.0, 1.0};
	double output[] = {1.0};
	int layer_num = 3;
	int neu_num[] = {2, 2, 1};*/
	vector<double> v_input(input, input+input_num);
	vector<double> v_output(output, output+output_num);
	vector<int> v_neu_num(neu_num, neu_num+layer_num);
	NN(layer_num, v_neu_num, v_input, v_output);
	getchar();
	return 0;
}

