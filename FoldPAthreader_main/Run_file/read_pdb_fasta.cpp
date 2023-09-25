#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char* argv[])
{
	if (argc != 3) {
        std::cout << "请提供pdb文件输入文件以及seq输出文件路径作为命令行参数" << std::endl;
        return 1;
    }

	const char* pdbfilePath = argv[1];
	const char* seqfilePath = argv[2];
	
	ifstream in(pdbfilePath);
	if (!in) {
		cout << "nonononono" << endl;
		return -1;
	}

	string line;
	string amino_acid;
	string chain = "0";
	string chain_temp = "0";
	string type = "000";
	string type_temp = "000";
	
	ofstream out(seqfilePath);

	while (getline(in, line)) {
		if (line.compare(0, 4, "ATOM") == 0) {
			chain = line.substr(21, 1);
			type = line.substr(23, 3);
			if (chain != chain_temp) {
				if (chain == "A") {
					out << ">seq " << chain << endl;
				}
				else {
					out << ">seq " << chain << endl;
				}
			}
			if (type != type_temp) {
				amino_acid = line.substr(17, 3);
				if (amino_acid == "ALA")
					out << "A";
				if (amino_acid == "CYS")
					out << "C";
				if (amino_acid == "ASP")
					out << "D";
				if (amino_acid == "GLU")
					out << "E";
				if (amino_acid == "PHE")
					out << "F";
				if (amino_acid == "GLY")
					out << "G";
				if (amino_acid == "HIS")
					out << "H";
				if (amino_acid == "ILE")
					out << "I";
				if (amino_acid == "LYS")
					out << "K";
				if (amino_acid == "LEU")
					out << "L";
				if (amino_acid == "MET")
					out << "M";
				if (amino_acid == "ASN")
					out << "N";
				if (amino_acid == "PRO")
					out << "P";
				if (amino_acid == "GLN")
					out << "Q";
				if (amino_acid == "ARG")
					out << "R";
				if (amino_acid == "SER")
					out << "S";
				if (amino_acid == "THR")
					out << "T";
				if (amino_acid == "VAL")
					out << "V";
				if (amino_acid == "TRP")
					out << "W";
				if (amino_acid == "TYR")
					out << "Y";
			}
		}
		type_temp = type;
		chain_temp = chain;
	}

	in.close();

}