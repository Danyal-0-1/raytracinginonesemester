#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>
#include <unordered_map>

#include "vec3.h"
#include "vec2.h"
#include "material.h"

// ============================================================
// Forward declarations
// ============================================================
struct Mesh;

// ============================================================
// MeshView — lightweight structure for passing mesh data to CUDA kernels
// ============================================================
struct MeshView
{
    Vec3*     positions;
    Vec3*     normals;
    Vec2*     uvs;
    uint32_t* indices;
    int32_t*  triangleObjIds;

    size_t numVertices;
    size_t numIndices;
    size_t numTriangles;

    HYBRID_FUNC Vec3     getVertex(uint32_t idx) const { return positions[idx]; }
    HYBRID_FUNC uint32_t getIndex(uint32_t idx)  const { return indices[idx]; }
};

// ============================================================
// Triangle — now carries per-vertex UVs
// ============================================================
struct Triangle {
    Vec3 v0, v1, v2;
    Vec3 n0, n1, n2;
    Vec2 uv0, uv1, uv2;   // <-- NEW: per-vertex texture coordinates

    HYBRID_FUNC Triangle()
        : v0(make_vec3(0,0,0)), v1(make_vec3(0,0,0)), v2(make_vec3(0,0,0)),
          n0(make_vec3(0,0,0)), n1(make_vec3(0,0,0)), n2(make_vec3(0,0,0)),
          uv0(), uv1(), uv2() {}

    HYBRID_FUNC Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : v0(a), v1(b), v2(c),
          n0(make_vec3(0,0,0)), n1(make_vec3(0,0,0)), n2(make_vec3(0,0,0)),
          uv0(), uv1(), uv2() {}

    HYBRID_FUNC Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
                         const Vec3& na, const Vec3& nb, const Vec3& nc)
        : v0(a), v1(b), v2(c), n0(na), n1(nb), n2(nc),
          uv0(), uv1(), uv2() {}

    HYBRID_FUNC Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
                         const Vec3& na, const Vec3& nb, const Vec3& nc,
                         const Vec2& ta, const Vec2& tb, const Vec2& tc)
        : v0(a), v1(b), v2(c), n0(na), n1(nb), n2(nc),
          uv0(ta), uv1(tb), uv2(tc) {}
};

// ============================================================
// HitRecord — now carries interpolated UVs
// ============================================================
struct HitRecord {
    int      triangleIdx;
    bool     hit;
    Vec3     p;
    Vec3     normal;
    double   t;
    bool     front_face;
    Material mat;
    Vec2     uv;          // interpolated texture coordinate at hit point
    bool     alpha_masked = false;  // set true by ApplyTextures when alpha < 0.5
};

struct MissRecord {
    Vec3 ray_dir;
    Vec3 color = make_vec3(0.0f, 0.0f, 0.0f);
};

// ============================================================
// Mesh — host-side mesh container
// ============================================================
struct Mesh
{
    std::vector<Vec3>     positions;
    std::vector<Vec3>     normals;
    std::vector<Vec2>     uvs;
    std::vector<uint32_t> indices;
    std::vector<int32_t>  triangleObjIds;

    bool hasNormals() const { return !normals.empty(); }
    bool hasUVs()     const { return !uvs.empty(); }

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

// ============================================================
// ParsedMTLMaterial — intermediate representation from .mtl file
// ============================================================
struct ParsedMTLMaterial {
    std::string name;
    Vec3  Kd        = make_vec3(0.8f, 0.8f, 0.8f);  // diffuse color
    Vec3  Ks        = make_vec3(0.0f, 0.0f, 0.0f);   // specular color
    Vec3  Ke        = make_vec3(0.0f, 0.0f, 0.0f);   // emission
    float Ns        = 32.0f;                           // specular exponent
    float d         = 1.0f;                            // dissolve (opacity)
    std::string map_Kd;    // diffuse texture path
    std::string map_d;     // alpha/dissolve mask (leaf cards, cutout geometry)
    std::string map_Bump;  // normal/bump map path (also handles "bump")
};

// ============================================================
// Vertex dedup helpers
// ============================================================
struct VertexKey
{
    int p = -1;
    int t = -1;
    int n = -1;
    bool operator==(const VertexKey& o) const { return p == o.p && t == o.t && n == o.n; }
};

struct VertexKeyHash
{
    size_t operator()(const VertexKey& k) const
    {
        size_t h1 = std::hash<int>()(k.p);
        size_t h2 = std::hash<int>()(k.t);
        size_t h3 = std::hash<int>()(k.n);
        size_t h = h1;
        h ^= (h2 + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        h ^= (h3 + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2));
        return h;
    }
};

// ============================================================
// Parsing utilities
// ============================================================
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
    while (*s >= '0' && *s <= '9') { v = v * 10 + (*s - '0'); ++s; }
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

static bool ParseFaceVertex(const char*& s, VertexKey& outKey,
                            size_t posCount, size_t uvCount, size_t nrmCount)
{
    int v = 0;
    if (!ParseInt(s, v)) return false;
    if (v < 0) outKey.p = (int)posCount + v;
    else       outKey.p = v - 1;
    outKey.t = -1;
    outKey.n = -1;

    if (*s != '/') return true;
    ++s;

    if (*s == '/') {
        ++s;
        int n = 0;
        if (!ParseInt(s, n)) return false;
        if (n < 0) outKey.n = (int)nrmCount + n; else outKey.n = n - 1;
        return true;
    }

    int t = 0;
    if (ParseInt(s, t)) {
        if (t < 0) outKey.t = (int)uvCount + t; else outKey.t = t - 1;
    }

    if (*s != '/') return true;
    ++s;

    int n = 0;
    if (ParseInt(s, n)) {
        if (n < 0) outKey.n = (int)nrmCount + n; else outKey.n = n - 1;
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
    if (it != dedup.end()) return it->second;

    uint32_t idx = (uint32_t)out.positions.size();
    dedup[key] = idx;

    out.positions.push_back(rawPos.at((size_t)key.p));

    if (wantUV) {
        Vec2 uv{};
        if (key.t >= 0 && key.t < (int)rawUV.size()) uv = rawUV[(size_t)key.t];
        out.uvs.push_back(uv);
    }

    if (wantNrm) {
        Vec3 n{};
        if (key.n >= 0 && key.n < (int)rawNrm.size()) n = rawNrm[(size_t)key.n];
        out.normals.push_back(n);
    }

    return idx;
}

// ============================================================
// Utility: extract directory from a file path
// ============================================================
static std::string OBJ_dirname(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

// ============================================================
// ParseMTLTexPath — extract the texture filename from an MTL
// texture line, skipping any Blender/OBJ option flags that
// precede it (e.g. "-s 17 17 1", "-bm 2.0", "-o 0 0 0").
// Per the MTL spec the actual filename is always the last
// whitespace-separated token on the line.
// ============================================================
static std::string ParseMTLTexPath(const char* s)
{
    std::string last, cur;
    while (*s && *s != '\n' && *s != '\r') {
        if (*s == ' ' || *s == '\t') {
            if (!cur.empty()) { last = std::move(cur); cur.clear(); }
        } else {
            cur.push_back(*s);
        }
        ++s;
    }
    if (!cur.empty()) last = std::move(cur);
    return last;
}

// ============================================================
// LoadMTL — parse a Wavefront .mtl file
// Returns a map from material name to ParsedMTLMaterial.
// ============================================================
inline std::unordered_map<std::string, ParsedMTLMaterial>
LoadMTL(const std::string& mtlPath, const std::string& baseDir)
{
    std::unordered_map<std::string, ParsedMTLMaterial> materials;

    FILE* f = std::fopen(mtlPath.c_str(), "rb");
    if (!f) {
        // Try relative to baseDir
        std::string alt = baseDir + "/" + mtlPath;
        f = std::fopen(alt.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "Warning: could not open MTL file '%s'\n", mtlPath.c_str());
            return materials;
        }
    }

    ParsedMTLMaterial* cur = nullptr;
    char line[1024];

    while (std::fgets(line, sizeof(line), f)) {
        const char* s = line;
        SkipWS(s);
        if (*s == '\0' || *s == '\n' || *s == '#') continue;

        // newmtl <name>
        if (std::strncmp(s, "newmtl", 6) == 0 && (s[6] == ' ' || s[6] == '\t')) {
            s += 6;
            SkipWS(s);
            // Extract name (rest of line, trimmed)
            std::string name;
            while (*s && *s != '\n' && *s != '\r') { name.push_back(*s); ++s; }
            // Trim trailing whitespace
            while (!name.empty() && (name.back() == ' ' || name.back() == '\t'))
                name.pop_back();
            materials[name] = ParsedMTLMaterial();
            materials[name].name = name;
            cur = &materials[name];
            continue;
        }

        if (!cur) continue;

        // Kd r g b
        if (s[0] == 'K' && s[1] == 'd' && (s[2] == ' ' || s[2] == '\t')) {
            s += 2;
            ParseFloat(s, cur->Kd.x);
            ParseFloat(s, cur->Kd.y);
            ParseFloat(s, cur->Kd.z);
            continue;
        }

        // Ks r g b
        if (s[0] == 'K' && s[1] == 's' && (s[2] == ' ' || s[2] == '\t')) {
            s += 2;
            ParseFloat(s, cur->Ks.x);
            ParseFloat(s, cur->Ks.y);
            ParseFloat(s, cur->Ks.z);
            continue;
        }

        // Ke r g b  (emission)
        if (s[0] == 'K' && s[1] == 'e' && (s[2] == ' ' || s[2] == '\t')) {
            s += 2;
            ParseFloat(s, cur->Ke.x);
            ParseFloat(s, cur->Ke.y);
            ParseFloat(s, cur->Ke.z);
            continue;
        }

        // Ns <float>
        if (s[0] == 'N' && s[1] == 's' && (s[2] == ' ' || s[2] == '\t')) {
            s += 2;
            ParseFloat(s, cur->Ns);
            continue;
        }

        // d <float>
        if (s[0] == 'd' && (s[1] == ' ' || s[1] == '\t')) {
            s += 1;
            ParseFloat(s, cur->d);
            continue;
        }

        // map_Kd [options] <path>
        if (std::strncmp(s, "map_Kd", 6) == 0 && (s[6] == ' ' || s[6] == '\t')) {
            s += 6; SkipWS(s);
            std::string texPath = ParseMTLTexPath(s);
            if (!texPath.empty() && texPath[0] != '/' && !(texPath.size() >= 2 && texPath[1] == ':'))
                texPath = baseDir + "/" + texPath;
            cur->map_Kd = texPath;
            continue;
        }

        // map_d [options] <path>  (alpha / dissolve mask for cutout geometry)
        if (std::strncmp(s, "map_d", 5) == 0 && (s[5] == ' ' || s[5] == '\t')) {
            s += 5; SkipWS(s);
            std::string texPath = ParseMTLTexPath(s);
            if (!texPath.empty() && texPath[0] != '/' && !(texPath.size() >= 2 && texPath[1] == ':'))
                texPath = baseDir + "/" + texPath;
            cur->map_d = texPath;
            continue;
        }

        // map_Bump [options] <path>  (normal map)
        if ((std::strncmp(s, "map_Bump", 8) == 0 && (s[8] == ' ' || s[8] == '\t')) ||
            (std::strncmp(s, "map_bump", 8) == 0 && (s[8] == ' ' || s[8] == '\t'))) {
            s += 8; SkipWS(s);
            std::string texPath = ParseMTLTexPath(s);
            if (!texPath.empty() && texPath[0] != '/' && !(texPath.size() >= 2 && texPath[1] == ':'))
                texPath = baseDir + "/" + texPath;
            cur->map_Bump = texPath;
            continue;
        }

        if (std::strncmp(s, "bump", 4) == 0 && (s[4] == ' ' || s[4] == '\t')) {
            s += 4; SkipWS(s);
            std::string texPath = ParseMTLTexPath(s);
            if (!texPath.empty() && texPath[0] != '/' && !(texPath.size() >= 2 && texPath[1] == ':'))
                texPath = baseDir + "/" + texPath;
            cur->map_Bump = texPath;
            continue;
        }

        // Ignore other directives (illum, Ni, Tr, etc.)
    }

    std::fclose(f);
    return materials;
}


// ============================================================
// LoadOBJ_ToMesh — updated to parse mtllib / usemtl
//
// New parameters:
//   outMtlMaterials — filled with per-object-id ParsedMTLMaterial
//                     (indexed by objectId - firstObjectId)
//   mtlLibName      — output: the mtllib filename found in the OBJ
// ============================================================
inline bool LoadOBJ_ToMesh(const std::string& path, Mesh& outMesh, int& nextObjectId,
                           std::vector<ParsedMTLMaterial>* outMtlMaterials = nullptr,
                           std::string* mtlLibName = nullptr)
{
    outMesh = Mesh{};

    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    const std::string baseDir = OBJ_dirname(path);

    std::vector<Vec3> rawPos;
    std::vector<Vec2> rawUV;
    std::vector<Vec3> rawNrm;

    bool fileHasUV  = false;
    bool fileHasNrm = false;

    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> dedup;
    dedup.reserve(10000);

    int currentObjId   = nextObjectId;
    int firstObjectId  = nextObjectId;
    bool firstTagFound = false;

    // MTL support
    std::unordered_map<std::string, ParsedMTLMaterial> mtlLib;
    std::string activeMtlName;
    // Map from objectId to the MTL material name active when that object was started
    std::unordered_map<int, std::string> objIdToMtlName;
    objIdToMtlName[currentObjId] = "";  // default

    char line[1024];

    while (std::fgets(line, sizeof(line), f))
    {
        const char* s = line;
        SkipWS(s);
        if (*s == '\0' || *s == '\n' || *s == '#') continue;

        // mtllib <filename>
        if (std::strncmp(s, "mtllib", 6) == 0 && (s[6] == ' ' || s[6] == '\t')) {
            s += 6;
            SkipWS(s);
            std::string mtlFile;
            while (*s && *s != '\n' && *s != '\r') { mtlFile.push_back(*s); ++s; }
            while (!mtlFile.empty() && (mtlFile.back() == ' ' || mtlFile.back() == '\t'))
                mtlFile.pop_back();
            if (mtlLibName) *mtlLibName = mtlFile;

            // Resolve path relative to OBJ directory
            std::string mtlPath = mtlFile;
            if (!mtlFile.empty() && mtlFile[0] != '/' && !(mtlFile.size() >= 2 && mtlFile[1] == ':'))
                mtlPath = baseDir + "/" + mtlFile;
            mtlLib = LoadMTL(mtlPath, baseDir);
            std::printf("  -> Loaded MTL with %zu materials\n", mtlLib.size());
            continue;
        }

        // usemtl <name>
        if (std::strncmp(s, "usemtl", 6) == 0 && (s[6] == ' ' || s[6] == '\t')) {
            s += 6;
            SkipWS(s);
            std::string mtlName;
            while (*s && *s != '\n' && *s != '\r') { mtlName.push_back(*s); ++s; }
            while (!mtlName.empty() && (mtlName.back() == ' ' || mtlName.back() == '\t'))
                mtlName.pop_back();
            activeMtlName = mtlName;

            // Each usemtl starts a new object group so materials are tracked per-object
            if (!outMesh.indices.empty()) {
                nextObjectId++;
                currentObjId = nextObjectId;
            }
            objIdToMtlName[currentObjId] = activeMtlName;
            continue;
        }

        // o objectname or g groupname
        if (*s == 'o' || *s == 'g')
        {
            char tag = *s;
            (void)tag;
            if (firstTagFound) {
                nextObjectId++;
                currentObjId = nextObjectId;
            } else {
                if (!outMesh.indices.empty()) {
                    nextObjectId++;
                    currentObjId = nextObjectId;
                }
                firstTagFound = true;
            }
            objIdToMtlName[currentObjId] = activeMtlName;
            continue;
        }

        // v x y z
        if (s[0] == 'v' && (s[1] == ' ' || s[1] == '\t')) {
            s += 1;
            Vec3 p{};
            if (!ParseFloat(s, p.x) || !ParseFloat(s, p.y) || !ParseFloat(s, p.z)) {
                std::fclose(f); return false;
            }
            rawPos.push_back(p);
            continue;
        }

        // vt u v
        if (s[0] == 'v' && s[1] == 't' && (s[2] == ' ' || s[2] == '\t')) {
            s += 2;
            Vec2 uv{};
            if (!ParseFloat(s, uv.x) || !ParseFloat(s, uv.y)) {
                std::fclose(f); return false;
            }
            rawUV.push_back(uv);
            fileHasUV = true;
            continue;
        }

        // vn x y z
        if (s[0] == 'v' && s[1] == 'n' && (s[2] == ' ' || s[2] == '\t')) {
            s += 2;
            Vec3 n{};
            if (!ParseFloat(s, n.x) || !ParseFloat(s, n.y) || !ParseFloat(s, n.z)) {
                std::fclose(f); return false;
            }
            rawNrm.push_back(n);
            fileHasNrm = true;
            continue;
        }

        // f ...
        if (s[0] == 'f' && (s[1] == ' ' || s[1] == '\t')) {
            s += 1;
            VertexKey keys[4];
            int count = 0;
            while (count < 4) {
                SkipWS(s);
                if (*s == '\0' || *s == '\n') break;
                VertexKey k{};
                if (!ParseFaceVertex(s, k, rawPos.size(), rawUV.size(), rawNrm.size())) break;
                if (k.t >= 0) fileHasUV = true;
                if (k.n >= 0) fileHasNrm = true;
                keys[count++] = k;
                while (*s != '\0' && *s != '\n' && *s != ' ' && *s != '\t') ++s;
            }
            if (count < 3) { std::fclose(f); return false; }

            uint32_t i0 = GetOrCreateVertex(keys[0], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);
            uint32_t i1 = GetOrCreateVertex(keys[1], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);
            uint32_t i2 = GetOrCreateVertex(keys[2], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);

            outMesh.indices.push_back(i0);
            outMesh.indices.push_back(i1);
            outMesh.indices.push_back(i2);
            outMesh.triangleObjIds.push_back(currentObjId);

            if (count == 4) {
                uint32_t i3 = GetOrCreateVertex(keys[3], rawPos, rawUV, rawNrm, outMesh, dedup, fileHasUV, fileHasNrm);
                outMesh.indices.push_back(i0);
                outMesh.indices.push_back(i2);
                outMesh.indices.push_back(i3);
                outMesh.triangleObjIds.push_back(currentObjId);
            }
            continue;
        }

        // Ignore other lines
    }

    std::fclose(f);

    if (outMesh.positions.empty() || outMesh.indices.empty()) return false;
    nextObjectId++;

    if (fileHasUV  && outMesh.uvs.size()     != outMesh.positions.size()) return false;
    if (fileHasNrm && outMesh.normals.size()  != outMesh.positions.size()) return false;

    // Build output MTL material list if requested
    if (outMtlMaterials) {
        int numObjects = nextObjectId - firstObjectId;
        outMtlMaterials->resize(numObjects);
        for (int oid = firstObjectId; oid < nextObjectId; ++oid) {
            int localIdx = oid - firstObjectId;
            auto nameIt = objIdToMtlName.find(oid);
            if (nameIt != objIdToMtlName.end() && !nameIt->second.empty()) {
                auto mtlIt = mtlLib.find(nameIt->second);
                if (mtlIt != mtlLib.end()) {
                    (*outMtlMaterials)[localIdx] = mtlIt->second;
                }
            }
        }
    }

    return true;
}

static void AppendMesh(Mesh& dst, const Mesh& src)
{
    uint32_t vertexOffset = (uint32_t)dst.positions.size();

    dst.positions.insert(dst.positions.end(), src.positions.begin(), src.positions.end());

    if (dst.hasNormals() || src.hasNormals()) {
        if (!dst.hasNormals() && !dst.positions.empty())
            dst.normals.resize(vertexOffset, Vec3{0,0,0});
        if (src.hasNormals())
            dst.normals.insert(dst.normals.end(), src.normals.begin(), src.normals.end());
        else
            dst.normals.resize(dst.normals.size() + src.positions.size(), Vec3{0,0,0});
    }

    if (dst.hasUVs() || src.hasUVs()) {
        if (!dst.hasUVs() && !dst.positions.empty())
            dst.uvs.resize(vertexOffset, Vec2{0,0});
        if (src.hasUVs())
            dst.uvs.insert(dst.uvs.end(), src.uvs.begin(), src.uvs.end());
        else
            dst.uvs.resize(dst.uvs.size() + src.positions.size(), Vec2{0,0});
    }

    size_t oldIndexCount = dst.indices.size();
    dst.indices.resize(oldIndexCount + src.indices.size());
    for (size_t i = 0; i < src.indices.size(); ++i) {
        dst.indices[oldIndexCount + i] = src.indices[i] + vertexOffset;
    }
    dst.triangleObjIds.insert(dst.triangleObjIds.end(),
                              src.triangleObjIds.begin(), src.triangleObjIds.end());
}