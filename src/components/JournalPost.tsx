import { motion, AnimatePresence } from "framer-motion";
import { X, Calendar, Clock, Share2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { JournalEntry } from "../types";

interface JournalPostProps {
    post: JournalEntry | null;
    onClose: () => void;
}

export default function JournalPost({ post, onClose }: JournalPostProps) {
    if (!post) return null;

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-[100] bg-bg overflow-y-auto"
            >
                {/* Header Navigation */}
                <div className="sticky top-0 z-50 bg-bg/80 backdrop-blur-md border-b border-stroke">
                    <div className="max-w-4xl mx-auto px-6 h-20 flex items-center justify-between">
                        <button
                            onClick={onClose}
                            className="flex items-center gap-2 text-sm text-muted hover:text-text-primary transition-colors"
                        >
                            <X className="w-4 h-4" />
                            <span>Close</span>
                        </button>

                        <div className="flex items-center gap-4">
                            <button className="p-2 rounded-full border border-stroke hover:bg-surface transition-colors">
                                <Share2 className="w-4 h-4 text-muted" />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Content */}
                <article className="max-w-3xl mx-auto px-6 py-16 md:py-24">
                    <motion.header
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="mb-12"
                    >
                        <div className="flex items-center gap-6 text-xs text-muted uppercase tracking-[0.2em] mb-6">
                            <div className="flex items-center gap-2">
                                <Calendar className="w-3 h-3" />
                                {post.date}
                            </div>
                            <div className="flex items-center gap-2">
                                <Clock className="w-3 h-3" />
                                {post.readTime}
                            </div>
                        </div>

                        <h1 className="text-4xl md:text-6xl lg:text-7xl font-display italic text-text-primary leading-[1.1] mb-8">
                            {post.title}
                        </h1>

                        <div className="w-full aspect-[16/9] rounded-[40px] overflow-hidden border border-stroke mb-16">
                            <img
                                src={post.image}
                                alt={post.title}
                                className="w-full h-full object-cover"
                                referrerPolicy="no-referrer"
                            />
                        </div>
                    </motion.header>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 }}
                        className="prose prose-invert prose-neutral max-w-none"
                    >
                        <div className="markdown-body">
                            <ReactMarkdown>{post.content}</ReactMarkdown>
                        </div>
                    </motion.div>

                    <footer className="mt-24 pt-12 border-t border-stroke flex flex-col items-center text-center">
                        <div className="w-16 h-16 rounded-full border border-stroke mb-6 overflow-hidden">
                            <img
                                src="https://picsum.photos/seed/michael/200/200"
                                alt="Michael Smith"
                                className="w-full h-full object-cover"
                                referrerPolicy="no-referrer"
                            />
                        </div>
                        <span className="text-sm text-muted mb-2">Written by</span>
                        <span className="text-lg font-display italic text-text-primary">Mustafa Bohra</span>
                        <p className="text-xs text-muted mt-4 max-w-xs">
                            Aspiring security engineer.
                        </p>
                    </footer>
                </article>
            </motion.div>
        </AnimatePresence>
    );
}
