//########################################################################
//### Program:   Warp Private Histogramming
//### Author:    C. Nugteren
//### Institute: Eindhoven University of Technology
//### Date:      22-02-2011
//### Version:   1.0
//### Filename:  bitmap.cu
//### Contents:  This file contains the BMP I/O functions. It includes a
//###            function to read BMP data to file. It also includes
//###            functions to load random and degenerate data.
//### Reference: http://parse.ele.tue.nl/
//########################################################################

//########################################################################
//### Includes
//########################################################################

#include <stdio.h>
#include <stdlib.h>
#include <config.h>
#pragma pack(1)

//########################################################################
//### Structures used in the BMP functions
//########################################################################

typedef struct {
	short type;
	int size;
	short reserved1;
	short reserved2;
	int offset;
} BMPHeader;
typedef struct {
	int size;
	int width;
	int height;
	short planes;
	short bitsPerPixel;
	unsigned compression;
	unsigned imageSize;
	int xPelsPerMeter;
	int yPelsPerMeter;
	int clrUsed;
	int clrImportant;
} BMPInfoHeader;

//########################################################################
//### Function to load BMP data from a file
//########################################################################

extern "C" unsigned char ** LoadBMPFile(uint *width, uint *height, const char* name);

extern "C" unsigned char * LoadBMPCustomDimensions(uint width, uint height, const char* name)
{
	uint sourceWidth, sourceHeight;

	unsigned char *indices = (unsigned char*) malloc(width * height * sizeof(unsigned char));
	unsigned char ** image = LoadBMPFile(&sourceWidth, &sourceHeight, name);
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			indices[y * width + x] = image[x % sourceWidth][y % sourceHeight];
		}
	}
	free(image[0]);
	free(image);
	return indices;
}


extern "C" unsigned char ** LoadBMPFile(uint *width, uint *height, const char* name)
{
	// Variable declarations
	BMPHeader hdr;
	BMPInfoHeader infoHdr;
	FILE *fd;
	uint i, y, x;

	// Open the file and scan the contents
	if(!(fd = fopen(name,"rb"))) { printf("***BMP load error: file access denied***\n"); exit(0);	}
	fread(&hdr, sizeof(hdr), 1, fd);
	if(hdr.type != 0x4D42) { printf("***BMP load error: bad file format***\n"); exit(0); }
	fread(&infoHdr, sizeof(infoHdr), 1, fd);
	if(infoHdr.bitsPerPixel != 24) { printf("***BMP load error: invalid color depth*** \n"); exit(0); }
	if(infoHdr.compression) { printf("***BMP load error: compressed image***\n"); exit(0); }
	(*width)  = infoHdr.width;
	(*height) = infoHdr.height;

	// Allocate memory to store the BMP's contents
	unsigned char ** image = (unsigned char **)malloc((*width) * sizeof(*image));
	unsigned char * image_1D = (unsigned char *)malloc((*width) * (*height) * sizeof(unsigned char));
	for(i=0; i<(*width); i++) {
		image[i] = &image_1D[i*(*height)];
	}

	// Read the BMP file and store the contents
	fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);
	for(y = 0; y < (*height); y++) {
		for(x = 0; x < (*width); x++)	{
			image[x][y] = (((unsigned char)fgetc(fd))+((unsigned char)fgetc(fd))+((unsigned char)fgetc(fd)))/3;
		}
		for(x = 0; x < (4 - (3 * (*width)) % 4) % 4; x++)	{
			fgetc(fd);
		}
	}

	// Exit the function and clean-up
	if(ferror(fd)) {
		printf("***Unknown BMP load error.***\n");
		free(image[0]);
		free(image);
		exit(0);
	}
	fclose(fd);
	return image;
}

//########################################################################
//### Function to load random data
//########################################################################

extern "C" unsigned char ** LoadRandomData(uint *width, uint *height)
{
	// Variable declarations
	uint x, y, i;

	// Allocate memory to store the BMP's contents
	unsigned char ** image = (unsigned char **)malloc((*width) * sizeof(*image));
	unsigned char * image_1D = (unsigned char *)malloc((*width) * (*height) * sizeof(unsigned char));
	for(i=0; i<(*width); i++)	{
		image[i] = &image_1D[i*(*height)];
	}

	// Initialize the random data
	srand(2011);

	// Store the random data to the array
	for(y = 0; y < (*height); y++) {
		for(x = 0; x < (*width); x++)	{
			image[x][y] = (unsigned char)(rand() % 256);
		}
	}
	return image;
}

//########################################################################
//### Function to load degenerate data
//########################################################################

extern "C" unsigned char ** LoadDegenerateData(uint *width, uint *height)
{
	// Variable declarations
	uint x, y, i;

	// Allocate memory to store the BMP's contents
	unsigned char ** image = (unsigned char **)malloc((*width) * sizeof(*image));
	unsigned char * image_1D = (unsigned char *)malloc((*width) * (*height) * sizeof(unsigned char));
	for(i=0; i<(*width); i++)	{
		image[i] = &image_1D[i*(*height)];
	}

	// Set one (random) degenerate value
	srand(2011);
	unsigned char value = (unsigned char)(rand() % 256);

	// Store the degenerate data to the array
	for(y = 0; y < (*height); y++) {
		for(x = 0; x < (*width); x++)	{
			image[x][y] = value;
		}
	}
	return image;
}

//########################################################################
