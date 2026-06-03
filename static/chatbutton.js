import React from "react";
import { motion } from "framer-motion";
import { BsRobot } from "react-icons/bs"; // AI Icon

const ChatButton = () => {
    const handleClick = () => {
        alert("Launching Sleep Apnea AI Chat...");
        // Connect this to your AI model API here
    };

    return (
        <motion.button
            className="chat-button"
            onClick={handleClick}
            initial={{ y: 10 }}
            animate={{ y: [10, 0, 10] }}
            transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut"
            }}
            whileHover={{ scale: 1.2, boxShadow: "0px 10px 25px rgba(0, 200, 255, 0.8)" }}
            whileTap={{ scale: 0.9 }}
        >
            <BsRobot size={30} />
        </motion.button>
    );
};

export default ChatButton;
