#include <Vector3.h>
#include "Atom.h"
#include "RigidTrans3.h"
#include "Matrix3.h"
#include "Molecule.h"
#include "PDB.h"
#include "Match.h"
#include "GeomHash.h"
#include "Triangle.h"
#include <chrono>
#include <iostream>

int main(int argc , char* argv[])
{
    if(argc != 4) {
        std::cerr << "Usage: "<<argv[0]<< "<epsilon> <Target_model_pdb_file> <text_file_of_pdb_files> " << std::endl;
        exit(1);
    }

    float epsilon = atof(argv[1]); // the largest allowed distance between two corresponding atoms in the alignment

    Molecule<Atom>  molTarget;

    // read target model
    std::ifstream fileTarget(argv[2]); // Target_model

    if(!fileTarget) {
        std::cout << "File " << argv[2] << "does not exist." << std::endl;
        return 0;
    }
    molTarget.readPDBfile(fileTarget, PDB::CAlphaSelector());

    // we insert the target molecule into hash
    // this will help us to find atoms that are close faster
    GeomHash <Vector3,int> gHash(3, epsilon); // 3 is a dimension and m_fDistThr is the size of the hash cube
    for(unsigned int i=0; i<molTarget.size(); i++) {
        gHash.insert(molTarget[i].position(), i); // coordinate is the key to the hash, we store atom index
    }

    // read pdb files from directory
    /* std::string path = argv[3];
    std::ifstream pdb_files;
    pdb_files.open(path);
    if (!pdb_files){
        std::cout << "Path " << argv[3] << "does not exist." << std::endl;
        return 0;
    }
    */
    std::string file;
    std::ofstream output;
    output.open("/cs/usr/talme/CLionProjects/hackathon1/pdb/output.csv");


    Molecule<Atom> molModel;
    std::ifstream fileModel(argv[3]);
    if(!fileModel) {
      std::cout << "File " << file << "does not exist." << std::endl;
      return 0;
    }

    while (molModel.readModelFromPDBfile(fileModel, PDB::CAlphaSelector()) > 0) { //(pdb_files >> file){

        // match is a class that stores the correspondence list, eg.
        // pairs of atoms, one from each molecule, that are matching
        Match match;

        for(unsigned int k=0; k < molModel.size(); k++) {
            Vector3 mol_atom = molModel[k].position();

            // find close target molecule atoms using the hash
            HashResult<int> result;
            gHash.query(mol_atom, epsilon, result); // key is mol atom coordinate

            // check if the atoms in the result are inside the distance threshold
            // the hash is a cube shape, there can be atoms further that the threshold
            for(auto x = result.begin(); x != result.end(); x++) {
                float dist = mol_atom.dist(molTarget[*x].position());
                if(dist <= epsilon) {
                    float score = (1 / (1 + dist));
                    match.add( *x , k, score, score);
                }
            }
            result.clear();
        }

        int correspond_size = match.size();
        std::cout << correspond_size << std::endl;
        output << correspond_size << ",";
        molModel.clear();
    }
    return 0;
}
