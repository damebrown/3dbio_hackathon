#include "Vector3.h"
#include "Atom.h"
#include "Molecule.h"
#include "PDB.h"
#include "Match.h"
#include "GeomHash.h"
#include <iostream>
#include <string>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

    //********Parameters********************
    float epsilon = atof(argv[3]); // distance threshold on atoms in correspondence

    std::cout << "Distance threshold: " << epsilon << std::endl;

    // read the two files into Molecule
    Molecule<Atom> mol1, mol2;
    // file to write the size of the correspondence between each two nanobodies.
    std::ofstream f("/cs/labs/dina/linoytsaban_14/correspond.csv");
    for (int i = 100; i < 200; i++) {
        std::string file1 = "/cs/labs/dina/linoytsaban_14/pdbs1/pdbs1/" + std::to_string(i) + ".pdb";
        mol1.readPDBfile(file1, PDB::CAlphaSelector());
        std::ifstream fileModel(file1);
        if(!fileModel) {
            std::cout << "FILE DOES NOT EXIST" << std::endl;
            break;
        }
        // next we insert the target molecule into hash
        // this will help us to find atoms that are close faster
        GeomHash<Vector3, int> gHash(3, epsilon); // 3 is a dimension and epsilon is the size of the
        // hash cube
        for (unsigned int k = 0; k < mol1.size(); k++) {
            gHash.insert(mol1[k].position(), k); // coordinate is the key to the hash, we store
            // atom index
        }
        for (int j = 100; j < 200; j++) {
        // this loop intentionaly starts from 100 and not 0 so we don't make the same calculation twice
            std::string file2 = "/cs/labs/dina/linoytsaban_14/pdbs1/pdbs1/" + std::to_string(j)
                    +".pdb";
            std::cout << file2 << std::endl;
            mol2.readPDBfile(file2, PDB::CAlphaSelector());
            std::ifstream fileModel1(file2);
            Match match;
            if(!fileModel1) {
                std::cout << "File " << file2 << "does not exist." << std::endl;
                break;
            }
            // match is a class that stores the correspondence list, eg.
            // pairs of atoms, one from each molecule, that are matching
            // add the pairs of atoms (one from target and one from model)
            // that are close enough to the match list
            for (unsigned int l = 0; l < mol2.size(); l++) {
                Vector3 mol_atom = mol2[l].position();
                // find close target molecule atoms using the hash
                HashResult<int> result;
                gHash.query(mol_atom, epsilon, result); // key is mol atom coordinate
                // check if the atoms in the result are inside the distance threshold
                // the hash is a cube shape, there can be atoms further that the threshold
                for (auto x = result.begin(); x != result.end(); x++) {
                    float dist = mol_atom.dist(mol1[*x].position());
                    if (dist <= epsilon) {
                        float score = (1 / (1 + dist));
                        match.add(*x, l, score, score);
                    }
                }
                result.clear();
            }

            int correspond_size = match.size();
            std::cout << correspond_size << std::endl;

            f << correspond_size << ",";
            mol2.clear();

        }

        f << std::endl;
        mol1.clear();
    }
    // by this point we have a triangular matrix that contains the size of match between
    // each two nanobodies in the given directory.
    //(triangular since we calculated the match for each pair once)
    f.close();
    return 0;

}

