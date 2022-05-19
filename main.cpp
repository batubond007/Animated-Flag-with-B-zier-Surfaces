#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <GL/glew.h>
//#include <OpenGL/gl3.h>   // The GL Header File
#include <GLFW/glfw3.h> // The GLFW header
#include <glm/glm.hpp> // GL Math library header
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> 
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define BUFFER_OFFSET(i) ((char*)NULL + (i))
#define PI 3.14159265359

using namespace std;

// Gloval Variables
float frameCount = 0;
float distortionAmount = 0.6f;
float distortionSpeed = -1.0f;
int SampleCount = 10;
int PatchCount = 1;

GLuint gProgram[2];
int gWidth, gHeight;

GLint modelingMatrixLoc[2];
GLint viewingMatrixLoc[2];
GLint projectionMatrixLoc[2];
GLint eyePosLoc[2];

glm::mat4 projectionMatrix;
glm::mat4 viewingMatrix;
glm::mat4 modelingMatrix;
glm::vec3 eyePos(0, 0, 0);

int activeProgramIndex = 1;

struct Face
{
    Face(int v[], int t[], int n[]) {
        vIndex[0] = v[0];
        vIndex[1] = v[1];
        vIndex[2] = v[2];
        tIndex[0] = t[0];
        tIndex[1] = t[1];
        tIndex[2] = t[2];
        nIndex[0] = n[0];
        nIndex[1] = n[1];
        nIndex[2] = n[2];
    }
    GLuint vIndex[3], tIndex[3], nIndex[3];
};

struct Vertex
{
    Vertex(GLfloat inX, GLfloat inY, GLfloat inZ) : x(inX), y(inY), z(inZ) { }
    GLfloat x, y, z;
    vector<Face> adjFaces;
};

struct Texture
{
    Texture(GLfloat inU, GLfloat inV) : u(inU), v(inV) { }
    GLfloat u, v;
};

struct Normal
{
    Normal(GLfloat inX, GLfloat inY, GLfloat inZ) : x(inX), y(inY), z(inZ) { }
    GLfloat x, y, z;
};

vector<Vertex> gVertices;
vector<Texture> gTextures;
vector<Normal> gNormals;
vector<Face> gFaces;

GLuint gVertexAttribBuffer, gIndexBuffer;
GLint gInVertexLoc, gInNormalLoc;
int gVertexDataSizeInBytes, gNormalDataSizeInBytes, gTexCoorDataSizeInBytes;



bool ParseObj(const string& fileName)
{
    fstream myfile;

    // Open the input 
    myfile.open(fileName.c_str(), std::ios::in);

    if (myfile.is_open())
    {
        string curLine;

        while (getline(myfile, curLine))
        {
            stringstream str(curLine);
            GLfloat c1, c2, c3;
            GLuint index[9];
            string tmp;

            if (curLine.length() >= 2)
            {
                if (curLine[0] == 'v')
                {
                    if (curLine[1] == 't') // texture
                    {
                        str >> tmp; // consume "vt"
                        str >> c1 >> c2;
                        gTextures.push_back(Texture(c1, c2));
                    }
                    else if (curLine[1] == 'n') // normal
                    {
                        str >> tmp; // consume "vn"
                        str >> c1 >> c2 >> c3;
                        gNormals.push_back(Normal(c1, c2, c3));
                    }
                    else // vertex
                    {
                        str >> tmp; // consume "v"
                        str >> c1 >> c2 >> c3;
                        gVertices.push_back(Vertex(c1, c2, c3));
                    }
                }
                else if (curLine[0] == 'f') // face
                {
                    str >> tmp; // consume "f"
                    char c;
                    int vIndex[3], nIndex[3], tIndex[3];
                    str >> vIndex[0]; str >> c >> c; // consume "//"
                    str >> nIndex[0];
                    str >> vIndex[1]; str >> c >> c; // consume "//"
                    str >> nIndex[1];
                    str >> vIndex[2]; str >> c >> c; // consume "//"
                    str >> nIndex[2];

                    assert(vIndex[0] == nIndex[0] &&
                        vIndex[1] == nIndex[1] &&
                        vIndex[2] == nIndex[2]); // a limitation for now

                 // make indices start from 0
                    for (int c = 0; c < 3; ++c)
                    {
                        vIndex[c] -= 1;
                        nIndex[c] -= 1;
                        tIndex[c] -= 1;
                    }

                    gFaces.push_back(Face(vIndex, tIndex, nIndex));
                }
                else
                {
                    cout << "Ignoring unidentified line in obj file: " << curLine << endl;
                }
            }

            //data += curLine;
            if (!myfile.eof())
            {
                //data += "\n";
            }
        }

        myfile.close();
    }
    else
    {
        return false;
    }

    assert(gVertices.size() == gNormals.size());

    return true;
}

bool ReadDataFromFile(
    const string& fileName, ///< [in]  Name of the shader file
    string& data)     ///< [out] The contents of the file
{
    fstream myfile;

    // Open the input 
    myfile.open(fileName.c_str(), std::ios::in);

    if (myfile.is_open())
    {
        string curLine;

        while (getline(myfile, curLine))
        {
            data += curLine;
            if (!myfile.eof())
            {
                data += "\n";
            }
        }

        myfile.close();
    }
    else
    {
        return false;
    }

    return true;
}

GLuint createVS(const char* shaderName)
{
    string shaderSource;

    string filename(shaderName);
    if (!ReadDataFromFile(filename, shaderSource))
    {
        cout << "Cannot find file name: " + filename << endl;
        exit(-1);
    }

    GLint length = shaderSource.length();
    const GLchar* shader = (const GLchar*)shaderSource.c_str();

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &shader, &length);
    glCompileShader(vs);

    char output[1024] = { 0 };
    glGetShaderInfoLog(vs, 1024, &length, output);
    //printf("VS compile log: %s\n", output);

    return vs;
}

GLuint createFS(const char* shaderName)
{
    string shaderSource;

    string filename(shaderName);
    if (!ReadDataFromFile(filename, shaderSource))
    {
        cout << "Cannot find file name: " + filename << endl;
        exit(-1);
    }

    GLint length = shaderSource.length();
    const GLchar* shader = (const GLchar*)shaderSource.c_str();

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &shader, &length);
    glCompileShader(fs);

    char output[1024] = { 0 };
    glGetShaderInfoLog(fs, 1024, &length, output);
    //printf("FS compile log: %s\n", output);

    return fs;
}

void initShaders()
{
    // Create the programs

    gProgram[0] = glCreateProgram();
    gProgram[1] = glCreateProgram();

    // Create the shaders for both programs

    GLuint vs1 = createVS("vert.glsl");
    GLuint fs1 = createFS("frag.glsl");

    GLuint vs2 = createVS("vert2.glsl");
    GLuint fs2 = createFS("frag2.glsl");

    // Attach the shaders to the programs

    glAttachShader(gProgram[0], vs1);
    glAttachShader(gProgram[0], fs1);

    glAttachShader(gProgram[1], vs2);
    glAttachShader(gProgram[1], fs2);

    // Link the programs

    glLinkProgram(gProgram[0]);
    GLint status;
    glGetProgramiv(gProgram[0], GL_LINK_STATUS, &status);

    if (status != GL_TRUE)
    {
        cout << "Program link failed" << endl;
        exit(-1);
    }

    glLinkProgram(gProgram[1]);
    glGetProgramiv(gProgram[1], GL_LINK_STATUS, &status);

    if (status != GL_TRUE)
    {
        cout << "Program link failed" << endl;
        exit(-1);
    }

    // Get the locations of the uniform variables from both programs

    for (int i = 0; i < 2; ++i)
    {
        modelingMatrixLoc[i] = glGetUniformLocation(gProgram[i], "modelingMatrix");
        viewingMatrixLoc[i] = glGetUniformLocation(gProgram[i], "viewingMatrix");
        projectionMatrixLoc[i] = glGetUniformLocation(gProgram[i], "projectionMatrix");
        eyePosLoc[i] = glGetUniformLocation(gProgram[i], "eyePos");
    }
}

void initVBO()
{
    GLuint vao;
    glGenVertexArrays(1, &vao);
    assert(vao > 0);
    glBindVertexArray(vao);
    //cout << "vao = " << vao << endl;

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    assert(glGetError() == GL_NONE);

    glGenBuffers(1, &gVertexAttribBuffer);
    glGenBuffers(1, &gIndexBuffer);

    assert(gVertexAttribBuffer > 0 && gIndexBuffer > 0);

    glBindBuffer(GL_ARRAY_BUFFER, gVertexAttribBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIndexBuffer);

    gVertexDataSizeInBytes = gVertices.size() * 3 * sizeof(GLfloat);
    gNormalDataSizeInBytes = gNormals.size() * 3 * sizeof(GLfloat);
    gTexCoorDataSizeInBytes = gTextures.size() * 2 * sizeof(GLfloat);
    int indexDataSizeInBytes = gFaces.size() * 3 * sizeof(GLuint);
    GLfloat* vertexData = new GLfloat[gVertices.size() * 3];
    GLfloat* normalData = new GLfloat[gNormals.size() * 3];
    GLfloat* texData = new GLfloat[gNormals.size() * 2];
    GLuint* indexData = new GLuint[gFaces.size() * 3];

    float minX = 1e6, maxX = -1e6;
    float minY = 1e6, maxY = -1e6;
    float minZ = 1e6, maxZ = -1e6;

    for (int i = 0; i < gVertices.size(); ++i)
    {
        vertexData[3 * i] = gVertices[i].x;
        vertexData[3 * i + 1] = gVertices[i].y;
        vertexData[3 * i + 2] = gVertices[i].z;

        minX = std::min(minX, gVertices[i].x);
        maxX = std::max(maxX, gVertices[i].x);
        minY = std::min(minY, gVertices[i].y);
        maxY = std::max(maxY, gVertices[i].y);
        minZ = std::min(minZ, gVertices[i].z);
        maxZ = std::max(maxZ, gVertices[i].z);
    }

    /*std::cout << "minX = " << minX << std::endl;
    std::cout << "maxX = " << maxX << std::endl;
    std::cout << "minY = " << minY << std::endl;
    std::cout << "maxY = " << maxY << std::endl;
    std::cout << "minZ = " << minZ << std::endl;
    std::cout << "maxZ = " << maxZ << std::endl;*/

    for (int i = 0; i < gNormals.size(); ++i)
    {
        normalData[3 * i] = gNormals[i].x;
        normalData[3 * i + 1] = gNormals[i].y;
        normalData[3 * i + 2] = gNormals[i].z;
    }

    for (int i = 0; i < gTextures.size(); i++)
    {
        texData[2 * i] = gTextures[i].u;
        texData[2 * i + 1] = gTextures[i].v;
    }

    for (int i = 0; i < gFaces.size(); ++i)
    {
        indexData[3 * i] = gFaces[i].vIndex[0];
        indexData[3 * i + 1] = gFaces[i].vIndex[1];
        indexData[3 * i + 2] = gFaces[i].vIndex[2];
    }


    glBufferData(GL_ARRAY_BUFFER, gVertexDataSizeInBytes + gNormalDataSizeInBytes + gTexCoorDataSizeInBytes, 0, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, gVertexDataSizeInBytes, vertexData);
    glBufferSubData(GL_ARRAY_BUFFER, gVertexDataSizeInBytes, gNormalDataSizeInBytes, normalData);
    glBufferSubData(GL_ARRAY_BUFFER, gVertexDataSizeInBytes + gNormalDataSizeInBytes, gTexCoorDataSizeInBytes, texData);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexDataSizeInBytes, indexData, GL_STATIC_DRAW);

    // done copying; can free now
    delete[] vertexData;
    delete[] normalData;
    delete[] texData;
    delete[] indexData;

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(gVertexDataSizeInBytes));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(gVertexDataSizeInBytes + gNormalDataSizeInBytes));
}

void initTexture() {
    int width, height, nrChannels;
    unsigned char* data = stbi_load("erhan.jpg", &width, &height, &nrChannels, 0);

    GLuint texture;
    glGenTextures(1, &texture);

    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    delete[]data;
}

float BarnsteinFunction(int i, float t) {
    float tmp = 1 - t;

    switch (i)
    {
        case 0:
            return powf(tmp, 3);
        case 1:
            return 3 * powf(tmp, 2) * t;
        case 2:
            return 3 * tmp * powf(t, 2);
        case 3:
            return powf(t, 3);
    }
}

glm::vec3 BezierSurfaceFunction(float u, float v, glm::vec3 **Points) {
    glm::vec3 result = glm::vec3(0, 0, 0);
    for (size_t i = 0; i <= 3; i++)
    {
        float B_i = BarnsteinFunction(i, u);
        for (size_t j = 0; j <= 3; j++)
        {
            float B_j = BarnsteinFunction(j, v);
            result += B_i * B_j * Points[j][i];
        }
    }
    return result;
}

// Sample Count 2 means there are no intermadiate points
void SampleVertices(int sampleCount, glm::vec3 Points[4][4]) {
    float step = 1.0f / (sampleCount - 1);

    for (int i = 0; i < sampleCount; i++)
    {
        float v = i * step;
        for (int j = 0; j < sampleCount; j++)
        {
            float u = j * step;
            //glm::vec3 P = BezierSurfaceFunction(u, v, Points);
           // gVertices.push_back(Vertex(P.x, P.y, P.z));
        }
    }
}

int GetSurfaceIndex(float value, int patchCount) {
    int i = 0;
    float step = 1.0f / patchCount;

    if (value == 0)
        return 0;
    else if (value == 1)
        return patchCount - 1;

    for (i = 0; i < patchCount; i++)
    {
        if (value <= (i+1)*step)
        {
            return i;
        }
    }

    return i;
}

// SP is Starting Point
void SampleVertices2(int sampleCount, int patchCount, glm::vec3 SP) {
    float patchStep = 1.0f / patchCount;
    
    glm::vec3 u(1, 0, 0);
    glm::vec3 v(0, 1, 0);
    glm::vec3 z(0, 0, 1);

    // Allocate space for surface control points
    glm::vec3 ****surfacePointsArray = new glm::vec3***[patchCount];
    for (int i = 0; i < patchCount; i++)
    {
        surfacePointsArray[i] = new glm::vec3 **[patchCount];
        for (int j = 0; j < patchCount; j++)
        {
            surfacePointsArray[i][j] = new glm::vec3*[4];
            for (int k = 0; k < 4; k++)
            {
                surfacePointsArray[i][j][k] = new glm::vec3[4];
            }
        }
    }

    int angle = frameCount; 
    float d1 = sin(angle * (PI / 180)) * distortionAmount;

    // Initilize surface control points
    for (int i = 0; i < patchCount; i++)    // in v direction
    {
        for (int j = 0; j < patchCount; j++)    // in u direction
        {
            glm::vec3 O1(0,0,0);
            glm::vec3 O2(0,0,0);
            glm::vec3 O3(0,0,0);
            glm::vec3 O4(0,0,0);

            O2.z = d1;
            O2.y = d1 * .5f;
            O3.z = -d1;
            O3.y = -d1 * .5f;
            if (j == patchCount - 1)
            {
                O4.z += d1 * 0.1f;
                O4.y += d1 * 0.2f;
            }

            glm::vec3 P11 = SP + v * (patchStep * i) + u * (patchStep * j) + O1;
            glm::vec3 P12 = SP + v * (patchStep * i) + u * (patchStep * (j + 0.33f)) + O2;
            glm::vec3 P13 = SP + v * (patchStep * i) + u * (patchStep * (j + 0.66f)) + O3;
            glm::vec3 P14 = SP + v * (patchStep * i) + u * (patchStep * (j + 1)) + O4;

            glm::vec3 P21 = SP + v * (patchStep * (i + 0.33f)) + u * (patchStep * j) + O1;
            glm::vec3 P22 = SP + v * (patchStep * (i + 0.33f)) + u * (patchStep * (j + 0.33f)) + O2;
            glm::vec3 P23 = SP + v * (patchStep * (i + 0.33f)) + u * (patchStep * (j + 0.66f)) + O3;
            glm::vec3 P24 = SP + v * (patchStep * (i + 0.33f)) + u * (patchStep * (j + 1)) + O4;

            glm::vec3 P31 = SP + v * (patchStep * (i + 0.66f)) + u * (patchStep * j) + O1;
            glm::vec3 P32 = SP + v * (patchStep * (i + 0.66f)) + u * (patchStep * (j + 0.33f)) + O2;
            glm::vec3 P33 = SP + v * (patchStep * (i + 0.66f)) + u * (patchStep * (j + 0.66f)) + O3;
            glm::vec3 P34 = SP + v * (patchStep * (i + 0.66f)) + u * (patchStep * (j + 1)) + O4;

            glm::vec3 P41 = SP + v * (patchStep * (i + 1)) + u * (patchStep * j) + O1;
            glm::vec3 P42 = SP + v * (patchStep * (i + 1)) + u * (patchStep * (j + 0.33f)) + O2;
            glm::vec3 P43 = SP + v * (patchStep * (i + 1)) + u * (patchStep * (j + 0.66f)) + O3;
            glm::vec3 P44 = SP + v * (patchStep * (i + 1)) + u * (patchStep * (j + 1)) + O4;


            glm::vec3 points[4][4] = {
                {P11, P12, P13, P14},
                {P21, P22, P23, P24},
                {P31, P32, P33, P34},
                {P41, P42, P43, P44}
            };  
            for (int x = 0; x < 4; x++)
            {
                for (int y = 0; y < 4; y++)
                {
                    surfacePointsArray[i][j][x][y] = points[x][y];
                }
            }
        }
    }

    // Sample surface
    float sampleStep = 1.0f / (sampleCount - 1);

    for (int i = 0; i < sampleCount; i++)
    {
        float v = i * sampleStep;
        for (int j = 0; j < sampleCount; j++)
        {
            float u = j * sampleStep;
            
            int u_index = GetSurfaceIndex(u, patchCount);
            int v_index = GetSurfaceIndex(v, patchCount);

            float u_arranged = (u - (1.0f / patchCount * u_index)) * patchCount;
            float v_arranged = (v - (1.0f / patchCount * v_index)) * patchCount;

            glm::vec3 P = BezierSurfaceFunction(u_arranged, v_arranged, surfacePointsArray[v_index][u_index]);
            gVertices.push_back(Vertex(P.x, P.y, P.z));

            //cout << "Value: " << u << "\tIndex: " << u_index << "\tArranged: " << u_arranged << "\tX: " << P.x << endl;
        }
    }

    for (int i = 0; i < patchCount; i++)
    {
        for (int j = 0; j < patchCount; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                delete [] surfacePointsArray[i][j][k];
            }
            delete[] surfacePointsArray[i][j];
        }
        delete[] surfacePointsArray[i];
    }
    delete[]surfacePointsArray;
}

void SampleNormals(int sampleCount) {
    int index = 0;
    for (int i = 0; i < sampleCount; i++)
    {
        for (int j = 0; j < sampleCount; j++, index++)
        {
            glm::vec3 normal(0, 0, 0);
            for (int l = 0; l < gVertices[index].adjFaces.size(); l++)
            {
                Face face = gVertices[index].adjFaces[l];
                
                if (face.vIndex[0] == index)
                {
                    glm::vec3 v1(gVertices[face.vIndex[1]].x - gVertices[face.vIndex[0]].x,
                        gVertices[face.vIndex[1]].y - gVertices[face.vIndex[0]].y,
                        gVertices[face.vIndex[1]].z - gVertices[face.vIndex[0]].z);
                    glm::vec3 v2(gVertices[face.vIndex[2]].x - gVertices[face.vIndex[0]].x,
                        gVertices[face.vIndex[2]].y - gVertices[face.vIndex[0]].y,
                        gVertices[face.vIndex[2]].z - gVertices[face.vIndex[0]].z);

                    normal += glm::cross(v1, v2);
                }
                else if (face.vIndex[1] == index)
                {
                    glm::vec3 v1(gVertices[face.vIndex[2]].x - gVertices[face.vIndex[1]].x,
                        gVertices[face.vIndex[2]].y - gVertices[face.vIndex[1]].y,
                        gVertices[face.vIndex[2]].z - gVertices[face.vIndex[1]].z);
                    glm::vec3 v2(gVertices[face.vIndex[0]].x - gVertices[face.vIndex[1]].x,
                        gVertices[face.vIndex[0]].y - gVertices[face.vIndex[1]].y,
                        gVertices[face.vIndex[0]].z - gVertices[face.vIndex[1]].z);

                    normal += glm::cross(v1, v2);
                }
                else if (face.vIndex[2] == index)
                {
                    glm::vec3 v1(gVertices[face.vIndex[0]].x - gVertices[face.vIndex[2]].x,
                        gVertices[face.vIndex[0]].y - gVertices[face.vIndex[2]].y,
                        gVertices[face.vIndex[0]].z - gVertices[face.vIndex[2]].z);
                    glm::vec3 v2(gVertices[face.vIndex[1]].x - gVertices[face.vIndex[2]].x,
                        gVertices[face.vIndex[1]].y - gVertices[face.vIndex[2]].y,
                        gVertices[face.vIndex[1]].z - gVertices[face.vIndex[2]].z);

                    normal += glm::cross(v1, v2);
                }
            }
            normal /= gVertices[index].adjFaces.size();
            //cout << "Indexes: " << i << ", " << j << "\tNormal: " << normal.r << "," << normal.g << "," << normal.b << endl;
            gNormals.push_back(Normal(normal.x, normal.y, normal.z));
        }
    }
}

void SampleTexCoors(int sampleCount) {
    float step = 1.0f / (sampleCount - 1);
    for (int i = 0; i < sampleCount; i++)
    {
        for (int j = 0; j < sampleCount; j++)
        {
            gTextures.push_back(Texture(j*step, 1-i*step));
        }
    }
}

void SampleFaces(int sampleCount){
    int index = 0;
    for (int v = 0; v < sampleCount - 1; v++)
    {
        index = v * sampleCount;
        for (int u = 0; u < sampleCount - 1; u++)
        {
            int vIndex_1[3] = { index + u, index + u + 1, index + u + sampleCount + 1 };
            int vIndex_2[3] = { index + u, index + u + sampleCount + 1, index + u + sampleCount };

            Face face_1(vIndex_1, vIndex_1, vIndex_1);
            gFaces.push_back(face_1);
            Face face_2(Face(vIndex_2, vIndex_2, vIndex_2));
            gFaces.push_back(face_2);
            
            gVertices[vIndex_1[0]].adjFaces.push_back(face_1);
            gVertices[vIndex_1[1]].adjFaces.push_back(face_1);
            gVertices[vIndex_1[2]].adjFaces.push_back(face_1);

            gVertices[vIndex_2[0]].adjFaces.push_back(face_2);
            gVertices[vIndex_2[1]].adjFaces.push_back(face_2);
            gVertices[vIndex_2[2]].adjFaces.push_back(face_2);
        }
    }
}

void CreateBezierSurface() {
    
    glm::vec3 StartingPoint = glm::vec3(0,0,0);

    SampleVertices2(SampleCount, PatchCount, StartingPoint);
    SampleFaces(SampleCount);
    SampleNormals(SampleCount);
    SampleTexCoors(SampleCount);
}

void ResetArrays() {
    for (int i = 0; i < gVertices.size(); i++)
    {
        gVertices[i].adjFaces.clear();
    }

    gVertices.clear();
    gTextures.clear();
    gNormals.clear();
    gFaces.clear();
}
void loop()
{
    ResetArrays();
    CreateBezierSurface();

    initVBO();
}

void drawModel()
{
    glBindBuffer(GL_ARRAY_BUFFER, gVertexAttribBuffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIndexBuffer);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(gVertexDataSizeInBytes));

    glDrawElements(GL_TRIANGLES, gFaces.size() * 3, GL_UNSIGNED_INT, 0);
    //glDrawElements(GL_LINES, gFaces.size() * 3, GL_UNSIGNED_INT, 0);
}

void display()
{
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0f);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    static float angle = 0;

    float angleRad = (float)(angle / 180.0) * M_PI;

    // Compute the modeling matrix

    //modelingMatrix = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, -0.4f, -5.0f));
    //modelingMatrix = glm::rotate(modelingMatrix, angleRad, glm::vec3(0.0, 1.0, 0.0));
    glm::mat4 matT = glm::translate(glm::mat4(1.0), glm::vec3(-0.5f, -0.4f, -5.0f));   // same as above but more clear

    modelingMatrix = matT;


    // Set the active program and the values of its uniform variables

    glUseProgram(gProgram[activeProgramIndex]);
    glUniformMatrix4fv(projectionMatrixLoc[activeProgramIndex], 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(viewingMatrixLoc[activeProgramIndex], 1, GL_FALSE, glm::value_ptr(viewingMatrix));
    glUniformMatrix4fv(modelingMatrixLoc[activeProgramIndex], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
    glUniform3fv(eyePosLoc[activeProgramIndex], 1, glm::value_ptr(eyePos));

    // Draw the scene
    drawModel();

    angle += 0.5;
}

void reshape(GLFWwindow* window, int w, int h)
{
    w = w < 1 ? 1 : w;
    h = h < 1 ? 1 : h;

    gWidth = w;
    gHeight = h;

    glViewport(0, 0, w, h);

    // Use perspective projection
    float fovyRad = (float)(45.0 / 180.0) * M_PI;
    projectionMatrix = glm::perspective(fovyRad, 1.0f, 1.0f, 100.0f);

    viewingMatrix = glm::mat4(1);
}

void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_G && action == GLFW_PRESS)
    {
        //glShadeModel(GL_SMOOTH);
        activeProgramIndex = 0;
    }
    else if (key == GLFW_KEY_P && action == GLFW_PRESS)
    {
        //glShadeModel(GL_SMOOTH);
        activeProgramIndex = 1;
    }
    else if (key == GLFW_KEY_F && action == GLFW_PRESS)
    {
        //glShadeModel(GL_FLAT);
    }
    // Sample Buttons
    else if (key == GLFW_KEY_W)
    {
        SampleCount++;
    }
    else if (key== GLFW_KEY_S)
    {
        if (SampleCount > 2)
        {
            SampleCount--;
        }
    }
    else if (key == GLFW_KEY_E)
    {
        PatchCount++;
    }
    else if (key == GLFW_KEY_D)
    {
        if (PatchCount > 1)
        {
            PatchCount--;
        }
    }
}

void mainLoop(GLFWwindow* window)
{
    while (!glfwWindowShouldClose(window))
    {
        frameCount += distortionSpeed;
        if (frameCount <= -360)
        {
            frameCount += 360;
        }

        loop();
        display();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

int main(int argc, char** argv)   // Create Main Function For Bringing It All Together
{
    GLFWwindow* window;
    if (!glfwInit())
    {
        exit(-1);
    }

    //glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    //glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    int width = 640, height = 480;
    window = glfwCreateWindow(width, height, "Simple Example", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize GLEW to setup the OpenGL Function pointers
    if (GLEW_OK != glewInit())
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return EXIT_FAILURE;
    }

    char rendererInfo[512] = { 0 };
    strcpy(rendererInfo, (const char*)glGetString(GL_RENDERER));
    strcat(rendererInfo, " - ");
    strcat(rendererInfo, (const char*)glGetString(GL_VERSION));
    glfwSetWindowTitle(window, rendererInfo);

    glEnable(GL_DEPTH_TEST);
    initShaders();
    initTexture();

    loop();

    glfwSetKeyCallback(window, keyboard);
    glfwSetWindowSizeCallback(window, reshape);

    reshape(window, width, height); // need to call this once ourselves
    mainLoop(window); // this does not return unless the window is closed

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
