"""
Cybersecurity Sample Data
Sample documents and queries for testing RAG systems
"""

# Sample cybersecurity documents
CYBERSECURITY_DOCUMENTS = [
    "Phishing is a type of social engineering attack where attackers send fraudulent emails or messages to trick users into revealing sensitive information such as passwords, credit card numbers, or personal data. Common signs of phishing include suspicious sender addresses, urgent requests, and links to fake websites.",
    
    "Ransomware is a type of malicious software that encrypts a victim's files and demands payment in cryptocurrency to restore access. Organizations should maintain regular backups, keep systems patched, and implement multi-factor authentication to protect against ransomware attacks.",
    
    "SQL Injection is a code injection technique that exploits vulnerabilities in database-driven applications. Attackers can insert malicious SQL statements into input fields to manipulate the database, potentially gaining unauthorized access to sensitive data. Input validation and parameterized queries are essential defenses.",
    
    "Multi-Factor Authentication (MFA) adds an extra layer of security by requiring users to provide two or more verification factors. This significantly reduces the risk of unauthorized access even if passwords are compromised. MFA is effective against phishing, credential stuffing, and brute force attacks.",
    
    "Cross-Site Scripting (XSS) vulnerabilities allow attackers to inject malicious scripts into web pages viewed by other users. These scripts can steal cookies, session tokens, or other sensitive information. Proper input validation and output encoding are crucial for preventing XSS attacks.",
    
    "A Distributed Denial of Service (DDoS) attack overwhelms a target system with a flood of traffic from multiple sources, making the service unavailable to legitimate users. Mitigation strategies include rate limiting, traffic filtering, and using content delivery networks.",
    
    "Zero-day vulnerabilities are security flaws that are unknown to the software vendor and have no available patch. Attackers can exploit these vulnerabilities before they are discovered and fixed. Organizations should implement defense-in-depth strategies and maintain incident response plans.",
    
    "Man-in-the-Middle (MitM) attacks occur when attackers intercept communication between two parties. This can happen on unsecured Wi-Fi networks or through compromised network infrastructure. Using encryption (HTTPS, VPNs) and verifying certificate authenticity helps prevent MitM attacks.",
    
    "Patch management is the process of regularly updating software to fix security vulnerabilities and bugs. Unpatched software is a common target for attackers. Organizations should maintain an inventory of systems, prioritize critical patches, and test updates before deployment.",
    
    "Security awareness training educates employees about cybersecurity threats and best practices. Human error is a leading cause of security breaches, so regular training on topics like phishing recognition, password hygiene, and data handling is essential.",
    
    "Buffer overflow vulnerabilities occur when a program writes more data to a buffer than it can hold, potentially allowing attackers to execute arbitrary code. Modern programming languages and compilers include protections, but legacy systems remain vulnerable.",
    
    "Intrusion Detection Systems (IDS) and Intrusion Prevention Systems (IPS) monitor network traffic for suspicious activity. IDS alerts administrators to potential threats, while IPS can automatically block malicious traffic. These tools are important components of network security.",
    
    "Encryption protects data confidentiality by converting information into an unreadable format. Strong encryption should be used for data at rest (stored data) and data in transit (network communications). Proper key management is crucial for maintaining encryption security.",
    
    "Weak authentication mechanisms, such as default passwords or single-factor authentication, create security vulnerabilities. Organizations should enforce strong password policies, implement MFA, and regularly audit authentication systems.",
    
    "Firewall systems act as barriers between trusted internal networks and untrusted external networks. They filter traffic based on predefined security rules, blocking unauthorized access while allowing legitimate communication. Properly configured firewalls are a fundamental security control."
]

# Sample test queries
TEST_QUERIES = [
    "What is phishing and how can I protect against it?",
    "How does ransomware work and what are the best defenses?",
    "What is SQL injection and how can it be prevented?",
    "Why is multi-factor authentication important?",
    "What are the best practices for preventing DDoS attacks?",
    "How can organizations protect against zero-day vulnerabilities?",
    "What is a man-in-the-middle attack?",
    "What are the key components of effective patch management?",
    "How does encryption help protect data?",
    "What mitigations are effective against malware?"
]

# Ground truth answers for evaluation
GROUND_TRUTH_ANSWERS = [
    "Phishing is a social engineering attack using fraudulent communications to steal sensitive information. Protection includes recognizing suspicious emails, verifying sender identities, using email filters, implementing security awareness training, and enabling multi-factor authentication.",
    
    "Ransomware encrypts victim files and demands payment for decryption. Best defenses include regular backups, keeping systems patched, implementing multi-factor authentication, network segmentation, and maintaining an incident response plan.",
    
    "SQL injection exploits database vulnerabilities by inserting malicious SQL code. Prevention requires input validation, parameterized queries, least privilege database access, regular security testing, and keeping database systems updated.",
    
    "Multi-factor authentication adds security layers beyond passwords, significantly reducing unauthorized access risk. It's effective against phishing, credential stuffing, and brute force attacks by requiring multiple verification factors.",
    
    "DDoS attack prevention includes implementing rate limiting, traffic filtering, using CDNs, maintaining excess bandwidth capacity, deploying IDS/IPS systems, and having an incident response plan.",
    
    "Protection against zero-day vulnerabilities requires defense-in-depth strategies, regular security monitoring, network segmentation, maintaining incident response plans, and staying informed about emerging threats through threat intelligence.",
    
    "Man-in-the-middle attacks intercept communications between parties. They occur on unsecured networks or through compromised infrastructure. Prevention includes using encryption (HTTPS, VPNs), verifying certificates, and avoiding unsecured public Wi-Fi.",
    
    "Effective patch management includes maintaining system inventories, prioritizing critical patches, testing updates before deployment, scheduling regular patch cycles, and monitoring for new vulnerabilities.",
    
    "Encryption converts data into unreadable format, protecting confidentiality. It should be used for data at rest and in transit, with strong algorithms and proper key management to ensure security.",
    
    "Malware mitigations include keeping software patched, using multi-factor authentication, implementing intrusion detection systems, maintaining regular backups, deploying firewalls, and providing security awareness training."
]


def get_sample_data():
    """
    Get sample cybersecurity data for testing.
    
    Returns:
        Dictionary with documents, queries, and ground truth
    """
    return {
        "documents": CYBERSECURITY_DOCUMENTS,
        "queries": TEST_QUERIES,
        "ground_truth": GROUND_TRUTH_ANSWERS
    }
