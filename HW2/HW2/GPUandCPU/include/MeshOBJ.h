#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <string>
#include <vector>
#include <unordered_map>

#include "vec3.h"
#include "material.h"


struct Vec2 { 
    float x = 0.0f; 
    float y = 0.0f; 

    Vec2() = default;
    Vec2(float _x, float _y) : x(_x), y(_y) {}
};

struct Mesh; // Forward decl

// Lightweight structure for passing mesh data to CUDA kernels
struct MeshView
{
    Vec3* positions;
    Vec3* normals; 
    Vec2* uvs;
    uint32_t* indices;
    int32_t* triangleObjIds;

    size_t numVertices;
    size_t numIndices;
    size_t numTriangles;

    // Optional: Device-friendly accessors
    HYBRID_FUNC Vec3 getVertex(uint32_t idx) const { return positions[idx]; }
    HYBRID_FUNC uint32_t getIndex(uint32_t idx) const { return indices[idx]; }
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 n0;
    Vec3 n1;
    Vec3 n2;

    HYBRID_FUNC Triangle()
        : v0(make_vec3(0.0f, 0.0f, 0.0f)),
          v1(make_vec3(0.0f, 0.0f, 0.0f)),
          v2(make_vec3(0.0f, 0.0f, 0.0f)),
          n0(make_vec3(0.0f, 0.0f, 0.0f)),
          n1(make_vec3(0.0f, 0.0f, 0.0f)),
          n2(make_vec3(0.0f, 0.0f, 0.0f)) {}

    HYBRID_FUNC Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : v0(a), v1(b), v2(c),
          n0(make_vec3(0.0f, 0.0f, 0.0f)),
          n1(make_vec3(0.0f, 0.0f, 0.0f)),
          n2(make_vec3(0.0f, 0.0f, 0.0f)) {}

    HYBRID_FUNC Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
                         const Vec3& na, const Vec3& nb, const Vec3& nc)
        : v0(a), v1(b), v2(c), n0(na), n1(nb), n2(nc) {}
};

struct HitRecord {
    int triangleIdx; // Index of the hit triangle
    bool hit;
    Vec3 p; // position of hit on ray
    Vec3 normal;
    double t; // parameter used to define point on ray direction of hit
    bool front_face;
    Material mat;
};

struct MissRecord {
    Vec3 ray_dir;
    Vec3 color = make_vec3(0.0f, 0.0f, 0.0f);
};

struct Mesh
{
    std::vector<Vec3> positions;   // per-vertex positions
    std::vector<Vec3> normals;     // normals (optional)
    std::vector<Vec2> uvs;         // uvs (optional)
    std::vector<uint32_t> indices; // triangles (3 per tri)
    std::vector<int32_t> triangleObjIds; // one ID per triangle

    bool hasNormals() const { return !normals.empty(); } // check if the mesh has normals
    bool hasUVs() const { return !uvs.empty(); } // check if the mesh has uvs

    // Helper to generate a view (Note: Pointers are valid only as long as vectors don't reallocate)
    MeshView getView() {
        return MeshView{
            positions.data(),
            normals.data(),
            uvs.data(),
            indices.data(),
            triangleObjIds.data(),
            positions.size(),
            indices.size(),
            indices.size() / 3
        };
    }
};

// Key representing a unique OBJ vertex reference (pos/uv/norm). Assuming zero-based for now.
struct VertexKey
{
    int p = -1; // position index (0-based)
    int t = -1; // texcoord index (0-based)
    int n = -1; // normal index (0-based)

    bool operator==(const VertexKey& o) const { return p == o.p && t == o.t && n == o.n; }
};

struct VertexKeyHash
{
    size_t operator()(const VertexKey& k) const
    {
        // Simple hash combine. (Speed up referencing/checking)
        size_t h1 = std::hash<int>()(k.p);
        size_t h2 = std::hash<int>()(k.t);
        size_t h3 = std::hash<int>()(k.n);
        size_t h = h1;
        h ^= (h2 + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        h ^= (h3 + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        return h;
    }
};

static void SkipWS(const char*& s)
{
    while (*s == ' ' || *s == '\t') ++s;
}

static bool ParseInt(const char*& s, int& out)
{
    SkipWS(s);

    bool neg = false;
    if (*s == '-') { neg = true; ++s; } 

    if (*s < '0' || *s > '9') return false;

    int v = 0;
    while (*s >= '0' && *s <= '9')
    {
        v = v * 10 + (*s - '0');
        ++s;
    }

    out = neg ? -v : v;
    return true;
}

static bool ParseFloat(const char*& s, float& out)
{
    SkipWS(s);
    char* endPtr = nullptr;
    out = std::strtof(s, &endPtr);
    if (endPtr == s) return false;
    s = endPtr;
    return true;
}

// Parse one face vertex token: "v/vt/vn", "v//vn", "v/vt", or "v"
// Supports negative indices (relative to current end of list) if lists are provided.
static bool ParseFaceVertex(const char*& s, VertexKey& outKey, size_t posCount, size_t uvCount, size_t nrmCount)
{
    int v = 0;
    if (!ParseInt(s, v)) return false;

    // Handle relative (negative) indices
    if (v < 0) outKey.p = (int)posCount + v;
    else       outKey.p = v - 1; // (OBJ is 1-based)

    outKey.t = -1;
    outKey.n = -1;

    if (*s != '/') return true; // only "v"

    ++s; // skip '/'

    if (*s == '/')
    {
        // "v//vn"
        ++s;
        int n = 0;
        if (!ParseInt(s, n)) return false;
        
        if (n < 0) outKey.n = (int)nrmCount + n;
        else       outKey.n = n - 1;
        return true;
    }

    // "v/vt" or "v/vt/vn"
    int t = 0;
    if (ParseInt(s, t)) {
        if (t < 0) outKey.t = (int)uvCount + t;
        else       outKey.t = t - 1;
    }

    if (*s != '/') return true;

    ++s; // skip '/'

    int n = 0;
    if (ParseInt(s, n)) {
        if (n < 0) outKey.n = (int)nrmCount + n;
        else       outKey.n = n - 1;
    }

    return true;
}


static uint32_t GetOrCreateVertex(
    const VertexKey& key,
    const std::vector<Vec3>& rawPos,
    const std::vector<Vec2>& rawUV,
    const std::vector<Vec3>& rawNrm,
    Mesh& out,
    std::unordered_map<VertexKey, uint32_t, VertexKeyHash>& dedup,
    bool wantUV,
    bool wantNrm)
{
    auto it = dedup.find(key);
    if (it != dedup.end())
        return it->second;

    uint32_t idx = (uint32_t)out.positions.size();
    dedup[key] = idx;

    // position required
    out.positions.push_back(rawPos.at((size_t)key.p));

    // optional streams: if file uses them, keep arrays aligned.
    if (wantUV)
    {
        Vec2 uv{};
        if (key.t >= 0 && key.t < (int)rawUV.size()) uv = rawUV[(size_t)key.t];
        out.uvs.push_back(uv);
    }

    if (wantNrm)
    {
        Vec3 n{};
        if (key.n >= 0 && key.n < (int)rawNrm.size()) n = rawNrm[(size_t)key.n];
        out.normals.push_back(n);
    }

    return idx;
}
    

inline bool LoadOBJ_ToMesh(const std::string& path, Mesh& outMesh, int& nextObjectId)
{
    outMesh = Mesh{}; // reset

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    std::vector<Vec3> rawPos;
    std::vector<Vec2> rawUV;
    std::vector<Vec3> rawNrm;

    bool fileHasUV = false;
    bool fileHasNrm = false;
    bool sawLineElement = false;

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> dedup;
    dedup.reserve(10000);

    // Identifying object ids
    int currentObjId = nextObjectId; 
    bool firstTagFound = false;

    char line[1024];

    while (std::fgets(line, sizeof(line), f))
    {
        const char* s = line;
        SkipWS(s);

        if (*s == '\0' || *s == '\n' || *s == '#')
            continue;
        
        // o objectname or g groupname
        if (*s == 'o' || *s == 'g')
        {
             // If we already parsed some faces for the 'current' object (implied or explicit),
             // then we must increment before starting the new one.
             // OR, simplistic approach: every 'o' starts a new ID.
             if (firstTagFound) {
                 nextObjectId++; // Use the next global ID
                 currentObjId = nextObjectId;
             } else {
                 // First tag found.
                 // If faces were already processed (default object), then this tag starts a 2nd object.
                 // If no faces yet, this tag names the 1st object (which uses incoming nextObjectId).
                 if (!outMesh.indices.empty()) {
                    nextObjectId++;
                    currentObjId = nextObjectId;
                 }
                 firstTagFound = true;
             }
             continue;
        }

        // v x y z
        if (s[0] == 'v' && (s[1] == ' ' || s[1] == '\t'))
        {
            s += 1;
            Vec3 p{};
            if (!ParseFloat(s, p.x) || !ParseFloat(s, p.y) || !ParseFloat(s, p.z))
            {
                std::fclose(f);
                return false;
            }
            rawPos.push_back(p);
            continue;
        }

        // vt u v
        if (s[0] == 'v' && s[1] == 't' && (s[2] == ' ' || s[2] == '\t'))
        {
            s += 2;
            Vec2 uv{};
            if (!ParseFloat(s, uv.x) || !ParseFloat(s, uv.y))
            {
                std::fclose(f);
                return false;
            }
            rawUV.push_back(uv);
            fileHasUV = true;
            continue;
        }

        // vn x y z
        if (s[0] == 'v' && s[1] == 'n' && (s[2] == ' ' || s[2] == '\t'))
        {
            s += 2;
            Vec3 n{};
            if (!ParseFloat(s, n.x) || !ParseFloat(s, n.y) || !ParseFloat(s, n.z))
            {
                std::fclose(f);
                return false;
            }
            rawNrm.push_back(n);
            fileHasNrm = true;
            continue;
        }

        // f ...
        if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t'))
        {
            s += 1;

            VertexKey keys[4];
            int count = 0;

            while (count < 4)
            {
                SkipWS(s);
                if (*s == '\0' || *s == '\n') break;

                VertexKey k{};
                // Passing current sizes for negative index resolution
                if (!ParseFaceVertex(s, k, rawPos.size(), rawUV.size(), rawNrm.size()))
                    break;

                if (k.t >= 0) fileHasUV = true;
                if (k.n >= 0) fileHasNrm = true;

                keys[count++] = k;

                // move to next token
                while (*s != '\0' && *s != '\n' && *s != ' ' && *s != '\t') ++s;
            }

            if (count < 3)
            {
                std::fclose(f);
                return false;
            }

            // create/reuse unified vertex indices
            uint32_t i0 = GetOrCreateVertex(keys[0], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);
            uint32_t i1 = GetOrCreateVertex(keys[1], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);
            uint32_t i2 = GetOrCreateVertex(keys[2], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);

            outMesh.indices.push_back(i0);
            outMesh.indices.push_back(i1);
            outMesh.indices.push_back(i2);
            outMesh.triangleObjIds.push_back(currentObjId);

            // quad -> two triangles (assuming we only handle triangles!!!)
            if (count == 4)
            {
                uint32_t i3 = GetOrCreateVertex(keys[3], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);
                outMesh.indices.push_back(i0);
                outMesh.indices.push_back(i2);
                outMesh.indices.push_back(i3);
                outMesh.triangleObjIds.push_back(currentObjId);
            }

            continue;
        }

        // l i j [k ...]
        // This loader is triangle-based; we record that line primitives exist and
        // provide a small fallback mesh later if no faces are present.
        if (s[0] == 'l' && (s[1] == ' ' || s[1] == '\t'))
        {
            sawLineElement = true;
            continue;
        }

        // ignore other lines (mtllib, usemtl, etc.)
    }

    std::fclose(f);

    // Fallback for wireframe-only OBJ files (e.g. BVH line visualizations):
    // build a simple cube from the first 8 vertices so the asset is still renderable.
    if (outMesh.indices.empty() && sawLineElement && rawPos.size() >= 8) {
        outMesh.positions.clear();
        outMesh.indices.clear();
        outMesh.triangleObjIds.clear();

        for (int i = 0; i < 8; ++i) {
            outMesh.positions.push_back(rawPos[static_cast<size_t>(i)]);
        }

        static const uint32_t kCubeIdx[] = {
            0, 1, 2, 0, 2, 3, // bottom
            4, 5, 6, 4, 6, 7, // top
            0, 1, 5, 0, 5, 4, // side
            1, 2, 6, 1, 6, 5, // side
            2, 3, 7, 2, 7, 6, // side
            3, 0, 4, 3, 4, 7  // side
        };
        const int triCount = static_cast<int>(sizeof(kCubeIdx) / (3 * sizeof(uint32_t)));
        outMesh.indices.insert(outMesh.indices.end(), kCubeIdx, kCubeIdx + sizeof(kCubeIdx) / sizeof(uint32_t));
        for (int t = 0; t < triCount; ++t) {
            outMesh.triangleObjIds.push_back(currentObjId);
        }
    }

    if (outMesh.positions.empty() || outMesh.indices.empty())
        return false;
    nextObjectId++; 


    if (fileHasUV && outMesh.uvs.size() != outMesh.positions.size()) return false;
    if (fileHasNrm && outMesh.normals.size() != outMesh.positions.size()) return false;

    return true;
}

static void AppendMesh(Mesh& dst, const Mesh& src)
{
    // The offset for indices
    uint32_t vertexOffset = (uint32_t)dst.positions.size();

    // Append vertices
    dst.positions.insert(dst.positions.end(), src.positions.begin(), src.positions.end());
    
    if (dst.hasNormals() || src.hasNormals()) {
        // If dst didn't have normals but src does, we need to pad dst????????? AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH 😵‍💫
        if (!dst.hasNormals() && !dst.positions.empty()) dst.normals.resize(vertexOffset, Vec3{0,0,0}); 
        
        if (src.hasNormals()) {
             dst.normals.insert(dst.normals.end(), src.normals.begin(), src.normals.end());
        } else {
             // src has no normals, append zeros
             dst.normals.resize(dst.normals.size() + src.positions.size(), Vec3{0,0,0});
        }
    }
    
    if (dst.hasUVs() || src.hasUVs()) {
        if (!dst.hasUVs() && !dst.positions.empty()) dst.uvs.resize(vertexOffset, Vec2{0,0});
        
        if (src.hasUVs()) {
             dst.uvs.insert(dst.uvs.end(), src.uvs.begin(), src.uvs.end());
        } else {
             dst.uvs.resize(dst.uvs.size() + src.positions.size(), Vec2{0,0});
        }
    }

    size_t oldIndexCount = dst.indices.size();
    dst.indices.resize(oldIndexCount + src.indices.size());
    for(size_t i = 0; i < src.indices.size(); ++i)
    {
        dst.indices[oldIndexCount + i] = src.indices[i] + vertexOffset;
    }
    dst.triangleObjIds.insert(dst.triangleObjIds.end(), src.triangleObjIds.begin(), src.triangleObjIds.end());
}
