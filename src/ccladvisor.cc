#include <cstdlib>
#include <iostream>
#include <fstream> // Include for std::ifstream

bool checkMSCCLXMLExists(const char* filePath) {
    std::ifstream file(filePath);
    return file.good();
}

int checkXmlFileExistence() {
    const char* xmlFilePath = std::getenv("MSCCL_XML_FILES");
    
    if (xmlFilePath != nullptr) {
        // If the environment variable is set, check if the file exists
        if (checkMSCCLXMLExists(xmlFilePath)) {
            std::cout << "The MSCCL XML file exists: " << xmlFilePath << std::endl;
            return 1;
        } else {
            std::cerr << "The MSCCL XML file does not exist: " << xmlFilePath << std::endl;
            return 0;
        }
    } else {
        // If the environment variable isn't set, handle the error
        std::cerr << "Error: MSCCL_XML_FILES environment variable is not set." << std::endl;
        return 0;
    }
}

void cclAdvisor() {
    // Attempt to get the environment variable
    const char* genXmlBashPath = std::getenv("GENMSCCLXML");
    
    if (genXmlBashPath != nullptr) {

        if (checkXmlFileExistence() == 1) {
            return;
        }
        // If the environment variable exists, use its value to build the command
        std::string command = "bash ";
        command += genXmlBashPath; // Append the path from the environment variable
        system(command.c_str());
    } else {
        // If the environment variable doesn't exist, handle the error
        std::cerr << "Error: GENMSCCLXML environment variable is not set." << std::endl;
    }
}
