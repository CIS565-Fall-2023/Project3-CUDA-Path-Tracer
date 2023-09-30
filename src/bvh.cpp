#pragma once
#include <numeric>
#include <cassert>
#include "scene.h"
#include "sceneStructs.h"
#include <fstream>

static int tnodeNum;

TBB::TBB() :min(glm::vec3(FLT_MAX)), max(glm::vec3(-FLT_MAX)) {}
TBB::TBB(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2) :min(glm::min(v0, glm::min(v1, v2))), max(glm::max(v0, glm::max(v1, v2))) {}
TBB::TBB(glm::vec3 min, glm::vec3 max) :min(min), max(max) {}

inline float TBB::area() const {
    glm::vec3 length = max - min;
    return length.x * length.y + length.y * length.z + length.z * length.x;
}

void sortAxis(std::vector<TriangleDetail>& triangles, std::vector<int>& obj_index, char axis, int li, int ri)
{
    int i = li;
    int j = ri;

    const float pivot = triangles[obj_index[(li + ri) / 2]].centroid[axis];
    while (true)
    {
        while (triangles[obj_index[i]].centroid[axis] < pivot) i++;
        while (triangles[obj_index[j]].centroid[axis] > pivot) j--;
        if (i >= j) break;
        std::swap(obj_index[i], obj_index[j]);
        i++; j--;
    }

    if (li < i - 1) sortAxis(triangles, obj_index, axis, li, i - 1);
    if (j + 1 < ri) sortAxis(triangles, obj_index, axis, j + 1, ri);
}

void printBVH(std::ofstream& fout, std::vector<TriangleDetail>& triangles, int index, int Level, std::vector<std::vector<TBVHNode>>& nodes)
{
    if (index < 0) return;

    for (int i = 0; i <= Level; i++)
    {
        fout << "  ";
    }

    fout << index << ": ";
    if (nodes[0][index].isLeaf)
    {
        fout << "leaf" << std::endl;
        for (int i = 0; i <= Level; i++)
        {
            fout << "  ";
        }
        fout << "triIdx:  " << nodes[0][index].triId << ", miss: " << nodes[0][index].miss << ", base: " << nodes[0][index].base << std::endl;
    }
    else
    {
        fout << "internal: ";
        fout << "triIdx:  " << nodes[0][index].triId << ", miss: " << nodes[0][index].miss << ", base: " << nodes[0][index].base << std::endl;
        printBVH(fout, triangles, nodes[0][index].left, Level + 1, nodes);
        printBVH(fout, triangles, nodes[0][index].right, Level + 1, nodes);
    }
}

int TBVH::splitBVH(std::vector<TriangleDetail>& triangles, std::vector<int> objIdx, int num, TBB& tbb, int face)
{
    std::vector<int> leftIndex, rightIndex;
    if (num <= 1) {
        const int id = tnodeNum;
        tnodeNum++;
        nodes[face][id].isLeaf = true;
        nodes[face][id].tbb = tbb;
        if (num == 0) {
            nodes[face][id].triId = -1;
        }
        else {
            nodes[face][id].triId = objIdx[0];
        }
        return id;
    }
    int axis = 0;
    int index = 0;
    float bestCost = FLT_MAX;
    TBB bestTBBLeft, bestTBBRight;

    if (num == 2) {
        leftIndex = { objIdx[0] };
        rightIndex = { objIdx[1] };
        bestTBBLeft = triangles[objIdx[0]].tbb;
        bestTBBRight = triangles[objIdx[1]].tbb;
    }
    else {
        std::vector<int> bestObjIdx;
        std::vector<float> leftArea(num);
        std::vector<TBB>leftTBBs(num);
        for (size_t i = 0; i < 3; i++)
        {
            auto tmp = objIdx;
            sortAxis(triangles, objIdx, i, 0, num - 1);

            float cost = 0.f;
            TBB tmpTBB;
            for (size_t j = 0; j < num; j++)
            {
                tmpTBB.expand(triangles[objIdx[j]].tbb);
                leftArea[j] = tmpTBB.area();
                leftTBBs[j] = tmpTBB;
            }
            tmpTBB = TBB();
            for (size_t j = num - 1; j > 0; j--)
            {
                tmpTBB.expand(triangles[objIdx[j]].tbb);
                const float tempCost = j * leftArea[j - 1] + (num - j) * tmpTBB.area();
                if (tempCost < bestCost) {
                    index = j - 1;
                    bestCost = tempCost;
                    axis = i;
                    bestTBBLeft = leftTBBs[index];
                    bestTBBRight = tmpTBB;
                }
            }
            if (axis == i) {
                bestObjIdx = objIdx;
            }
        }
        leftIndex.assign(bestObjIdx.begin(), bestObjIdx.begin() + index + 1);
        rightIndex.assign(bestObjIdx.begin() + index + 1, bestObjIdx.end());
    }

    const int id = tnodeNum;
    tnodeNum++;
    nodes[face][id].tbb = tbb;
    nodes[face][id].isLeaf = false;

    if (bestTBBLeft.min.x < bestTBBRight.min.x)
    {
        nodes[face][id].left = splitBVH(triangles, leftIndex, index + 1, bestTBBLeft, face);
        nodes[face][id].right = splitBVH(triangles, rightIndex, num - (index + 1), bestTBBRight, face);
    }
    else
    {
        nodes[face][id].left = splitBVH(triangles, rightIndex, num - (index + 1), bestTBBRight, face);
        nodes[face][id].right = splitBVH(triangles, leftIndex, index + 1, bestTBBLeft, face);
    }
    return id;
}



void TBVH::reorderNodes(std::vector<TriangleDetail>& triangles, int face, int index)
{
    if (index < 0) return;
    if (tnodeNum == triangles.size() * 2) return;

    int id = tnodeNum;
    tnodeNum++;
    nodes[face][id] = nodes[6][index];
    nodes[face][id].base = index;
    if (nodes[6][index].isLeaf) return;
    reorderNodes(triangles, face, nodes[6][index].left);
    reorderNodes(triangles, face, nodes[6][index].right);
}


int TBVH::reorderTree(std::vector<TriangleDetail>& triangles, int face, int index)
{
    if (nodes[6][index].isLeaf)
    {
        tnodeNum++;
        return tnodeNum - 1;
    }
    int id = tnodeNum;
    tnodeNum++;
    nodes[face][id].left = reorderTree(triangles, face, nodes[6][index].left);
    nodes[face][id].right = reorderTree(triangles, face, nodes[6][index].right);
    return id;
}


void TBVH::setLeftMiss(int id, int idParent, int face)
{
    if (nodes[face][id].isLeaf)
    {
        nodes[face][id].miss = id + 1;
        return;
    }
    nodes[face][id].miss = nodes[face][idParent].right;
    setLeftMiss(nodes[face][id].left, id, face);
    setLeftMiss(nodes[face][id].right, id, face);
}


void TBVH::setRightMiss(int id, int idParent, int face)
{
    if (nodes[face][id].isLeaf)
    {
        nodes[face][id].miss = id + 1;
        return;
    }
    if (nodes[face][idParent].right == id)
        nodes[face][id].miss = nodes[face][idParent].miss;
    setRightMiss(nodes[face][id].left, id, face);
    setRightMiss(nodes[face][id].right, id, face);
}


TBVH::TBVH(std::vector<TriangleDetail>& triangles, TBB& tbb)
    :nodes(std::vector<std::vector<TBVHNode>>(7, std::vector<TBVHNode>(triangles.size() * 2)))
{
    for (int face = 0; face <= 5; face++)
    {
        const int num = triangles.size();
        std::vector<int>objIdx(num);
        std::iota(objIdx.begin(), objIdx.end(), 0);
        tnodeNum = 0;
        for (int i = 0; i < num * 2; i++) {
            nodes[face][i].miss = -1;
            nodes[face][i].base = i;
        }

        if (face == 0) {
            splitBVH(triangles, objIdx, num, tbb, face);
            nodesNum = tnodeNum;
            std::for_each(nodes[6].begin(), nodes[6].end(), [](TBVHNode& node) {node.miss = -1; });
        }
        else {
            nodes[6] = nodes[0];
            for (int i = 0; i <= nodesNum - 1; i++) {
                if (nodes[6][i].isLeaf) continue;
                const auto& tbbLeft = nodes[6][nodes[6][i].left].tbb;
                const auto& tbbRight = nodes[6][nodes[6][i].right].tbb;
                if ((face == 1) && (tbbLeft.max.x > tbbRight.max.x)) continue;
                if ((face == 2) && (tbbLeft.min.y < tbbRight.min.y)) continue;
                if ((face == 3) && (tbbLeft.max.y > tbbRight.max.y)) continue;
                if ((face == 4) && (tbbLeft.min.z < tbbRight.min.z)) continue;
                if ((face == 5) && (tbbLeft.max.z > tbbRight.max.z)) continue;
                std::swap(nodes[6][i].left, nodes[6][i].right);
            }

            tnodeNum = 0;
            reorderNodes(triangles, face, 0);
            tnodeNum = 0;
            reorderTree(triangles, face, 0);
        }

        nodes[face][0].miss = -1;
        setLeftMiss(0, 0, face);
        nodes[face][0].miss = -1;
        setRightMiss(0, 0, face);
        nodes[face][0].miss = -1;
    }
    std::ofstream fout("../bvh.txt", std::ios::out | std::ios::trunc);
    printBVH(fout, triangles, 0, 0, nodes);
    fout.close();
}
