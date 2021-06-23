#include "Vector3.h"
#include "Atom.h"
#include "RigidTrans3.h"
#include "Matrix3.h"
#include "Molecule.h"
#include "PDB.h"
#include "Match.h"
#include "GeomHash.h"
#include "Triangle.h"
#include <iostream>
#include <fstream>

int isRNA(char* path)
{
    std::ifstream tempFile(path);
    Molecule<Atom> tempMol;
    tempMol.readPDBfile(tempFile, PDB::PSelector());
    if(tempMol.size() == 0){
        return 0;
    }
    return 1;
}

void compute(Match &match, GeomHash<Vector3, int> &gHash, float epsilon, Molecule<Atom> &molModel,
             Molecule<Atom> &molTarget,
             RigidTrans3 &tempTrans) {
// apply rotation on each atom in the model molecule and
    // add the pairs of atoms (one from target and one from model)
    // that are close enough to the match list
    for (unsigned int i = 0; i < molModel.size(); i++) {
        Vector3 mol_atom = tempTrans * molModel[i].position(); // rotate

        // find close target molecule atoms using the hash
        HashResult<int> result;
        gHash.query(mol_atom, epsilon, result); // key is mol atom coordinate

        // check if the atoms in the result are inside the distance threshold
        // the hash is a cube shape, there can be atoms further that the threshold
        for (auto x = result.begin(); x != result.end(); x++) {
            float dist = mol_atom.dist(molTarget[*x].position());
            if (dist <= epsilon) {
                float score = (1 / (1 + dist));
                match.add(*x, i, score, score);
            }
        }
        result.clear();
    }
}

int nomain(int argc , char* argv[]) {

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << "epsilon target.pdb model.pdb" << std::endl;
        exit(1);
    }

    //********Parameters********************
    float epsilon = atof(argv[1]); // distance threshold on atoms in correspondence
    // We assume that epsilon is larger than zero:
    if (!epsilon) {
        std::cout << argv[1] << " invalid epsilon value" << std::endl;
        return 0;
    }

    // read the two files into Molecule
    Molecule<Atom> molModel, molTarget;
    std::ifstream fileModel(argv[3]);
    std::ifstream fileTarget(argv[2]);

    if (!fileModel) {
        std::cout << "File " << argv[2] << "does not exist." << std::endl;
        return 0;
    }
    if (!fileTarget) {
        std::cout << "File " << argv[3] << "does not exist." << std::endl;
        return 0;
    }

    if (!isRNA(argv[3])) {
        molModel.readPDBfile(fileModel, PDB::CAlphaSelector());
        molTarget.readPDBfile(fileTarget, PDB::CAlphaSelector());
    } else {
        molModel.readPDBfile(fileModel, PDB::PSelector());
        molTarget.readPDBfile(fileTarget, PDB::PSelector());
    }

    // calculate center of mass
    Vector3 vectModelMass(0, 0, 0);
    for (unsigned int i = 0; i < molModel.size(); i++) {
        vectModelMass += molModel[i].position();
    }
    vectModelMass /= molModel.size();

    Vector3 vectTargetMass(0, 0, 0);
    for (unsigned int i = 0; i < molTarget.size(); i++) {
        vectTargetMass += molTarget[i].position();
    }
    vectTargetMass /= molTarget.size();

    // transform the molecules to the center of the coordinate system
    molModel += (-vectModelMass);
    molTarget += (-vectTargetMass);

    // next we insert the target molecule into hash
    // this will help us to find atoms that are close faster
    GeomHash<Vector3, int> gHash(3,
                                 epsilon); // 3 is a dimension and m_fDistThr is the size of the hash cube
    for (unsigned int i = 0; i < molTarget.size(); i++) {
        gHash.insert(molTarget[i].position(),
                     i); // coordinate is the key to the hash, we store atom index
    }

    // now we try random rotations and choose the best alignment from random rotations
    unsigned int iMaxSize = 0;
    RigidTrans3 rtransBest, tempTrans;
    float rmsd = 0;
    for (long unsigned int iTarget = 0; iTarget < molTarget.size() - 2; iTarget++) {
        Triangle targetTriangle = Triangle(molTarget[iTarget].position(),
                                           molTarget[iTarget + 1].position(),
                                           molTarget[iTarget + 2].position());
        std::cout << iTarget << " / " << molTarget.size() << std::endl;
        for (long unsigned int iModle = 0; iModle < molModel.size() - 2; iModle++) {
            Triangle modleTriangle = Triangle(molModel[iModle].position(),
                                              molModel[iModle + 1].position(),
                                              molModel[iModle + 2].position());
            tempTrans = targetTriangle | modleTriangle;

            // match is a class that stores the correspondence list, eg.
            // pairs of atoms, one from each molecule, that are matching
            Match match;
            compute(match, gHash, epsilon, molModel, molTarget, tempTrans);
            //calculates transformation that is a little better than "rotation"
            match.calculateBestFit(molTarget, molModel);

            if (iMaxSize < match.size()) {
                iMaxSize = match.size();
                rtransBest = match.rigidTrans();
                rmsd = match.rmsd();
            }
        }
    }

    // main:
    // std::cout << "Max Alignment Size: " << iMaxSize << std::endl;
    // std::cout << "Rigid Trans: " << rtransBest << std::endl;

    // std::chrono::duration<double> elapsed_seconds = end-start;
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
    // std::cout << "rmsd: " << rmsd << std::endl;
    // // Save to pdb:
    std::cout << iMaxSize << " " << rmsd << " " << rtransBest << " " << std::endl;
    molModel *= rtransBest;
    std::fstream out("transformed.pdb");
    out << molModel;
    return 0;
}
