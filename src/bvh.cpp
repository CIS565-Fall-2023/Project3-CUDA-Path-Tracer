#include "bvh.h"
#include <iostream>

using namespace std;

//BVH::BVH()
//	//:orderedTris(), root(nullptr), rootNodeIdx(0), nodes()
//{}

/// <summary>
/// Recursively builds a BVH by splitting triangles along the longest axis.
/// BVH is built in a depth-first fashion. The vector of BVHNodes is a linearly compacted vector such that this logic from PBRT is followed: https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies#CompactBVHForTraversal
/// </summary>
/// <param name="totalNodes">Total nodes so far in the overall bvhNodes array</param>
/// <param name="startTriIdx">Pass 0 here. Used internally in recursive calls.</param>
/// <param name="nTris">Total tris in consideration for this BVH.</param>
/// <param name="tris">Full triangle array of all meshes</param>
/// <param name="bvhNodes">Full BVHNodes array of all meshes. This is updated while building the BVH with new BVHNodes</param>
/// <returns>Index of this BVH node in the bvhNodes vector</returns>
int buildBVHRecursively(int& totalNodes, int startOffset, int nTris, const std::vector<Triangle>& tris, std::vector<int>& triIndices, std::vector<BVHNode>& bvhNodes)
{
	// Compute AABB of all tris
	AABB aabb;
	for (int i = startOffset; i < startOffset + nTris; i++)
	{
		aabb = AABB::combine(aabb, tris[triIndices[i]].aabb);
	}

	// Init new node
	//uPtr<BVHNode> uNode = mkU<BVHNode>();
	BVHNode node = BVHNode();// uNode.get();
	int nodeIndex = totalNodes;
	totalNodes++;
	bvhNodes.push_back(std::move(node));

	if (nTris == 1)
	{
		// base case
		//int firstTriIdx = orderedTris.size();
		//orderedTris.push_back(&tris[startTriIdx]);
		node.initAsLeafNode(triIndices[startOffset], aabb);
	}
	else
	{
		// General case
		int splitAxis = aabb.getLongestSplitAxis();

		// Compute centroid bounds
		AABB centroidAABB;
		for (int i = startOffset; i < startOffset + nTris; i++)
		{
			centroidAABB.include(tris[triIndices[i]].centroid);
		}
		int dimToSortOn = centroidAABB.getLongestSplitAxis();

		// Sort along longest axis

		// we're simply going to sort the indices instead of sorting the triangles
		// That way we don't mess with the indices of the triangles
		// and we can directly generate the linear compacted representation of the bvh in a single go
		std::sort(triIndices.begin() + startOffset, triIndices.begin() + startOffset + nTris, sortTriIndicesBasedOnDim(tris, dimToSortOn));
		
		// Split the sorted vecs from the middle
		int half = nTris / 2;
		int mid = startOffset + half;
		int end = std::max(0, nTris - mid);

		//cout << "left: " << startOffset << ", mid: " << mid << ", end: " << mid + nTris - half << endl;
		int leftChildIdx = buildBVHRecursively(totalNodes, startOffset, half, tris, triIndices, bvhNodes);
		int rightChildIdx = buildBVHRecursively(totalNodes, mid, nTris - half, tris, triIndices, bvhNodes);
		node.initInterior(splitAxis, leftChildIdx, rightChildIdx, bvhNodes);
	}

	return nodeIndex;
}

//const BVHNode* BVH::getRootNode() const
//{
//	return root;
//}