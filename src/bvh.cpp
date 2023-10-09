#include "bvh.h"
#include <iostream>

using namespace std;

BVH::BVH()
	:orderedTris(), root(nullptr), rootNodeIdx(0), nodes()
{}

/// <summary>
/// Returns the index of this BVH node in the bvhNodes vector
/// </summary>
/// <param name="totalNodes"></param>
/// <param name="startTriIdx"></param>
/// <param name="endTriIdx"></param>
/// <param name="tris"></param>
/// <param name="bvhNodes"></param>
/// <returns></returns>
int BVH::buildBVHRecursively(int& totalNodes, int startOffset, int nTris, const std::vector<Triangle>& tris, std::vector<int>& triIndices, std::vector<BVHNode>& bvhNodes)
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
	node.nodeIdx = totalNodes++;
	bvhNodes.push_back(std::move(node));

	if (nTris == 1)
	{
		// base case
		int firstTriIdx = orderedTris.size();
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

	return node.nodeIdx;
}

const BVHNode* BVH::getRootNode() const
{
	return root;
}