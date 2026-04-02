import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { JournalEntry } from "../types";
import JournalPost from "./JournalPost";

const entries: JournalEntry[] = [
  {
    id: "motion-design",
    title: "A Practical Guide to False Data Injection in Islanded AC Microgrids",
    date: "Mar 24, 2026",
    readTime: "5 min read",
    image: "https://i1.rgstatic.net/publication/335321361_On_Detection_of_False_Data_in_Cooperative_DC_Microgrids-A_Discordant_Element_Approach/links/5d5e7282299bf1b97cfd6071/largepreview.png",
    content: `
# The Future of Motion Design

Motion design is no longer just an afterthought in digital product development. It has become a core component of the user experience, bridging the gap between static interfaces and human-like interaction.

## Beyond Aesthetics

In the coming years, we'll see motion design move beyond simple transitions and animations. It will become a tool for:

1. **Contextual Awareness**: Helping users understand where they are and where they're going.
2. **Emotional Connection**: Creating a sense of delight and personality in otherwise sterile environments.
3. **Cognitive Load Reduction**: Guiding the user's eye and highlighting important information.

## The Rise of Generative Motion

With the advent of AI and generative tools, motion design will become more dynamic and personalized. Imagine an interface that adapts its animation speed and style based on the user's behavior and preferences.

> "Motion design is the soul of the interface."

As we continue to push the boundaries of what's possible, the key will be to maintain a balance between innovation and usability. Motion should always serve a purpose, never distract.
    `,
  },
  {
    id: "minimalism",
    title: "How I Automated 80% of SOC Triage with Tines & LimaCharlie",
    date: "Feb 12, 2026",
    readTime: "8 min read",
    image: "https://www.runtime.news/content/images/size/w1200/2025/03/Screenshot-2025-03-27-at-3.00.54-PM.png",
    content: `
# Minimalism in Digital Products

Minimalism is not just about removing elements; it's about focusing on what's truly important. In a world of constant distraction, a minimalist approach can be a breath of fresh air for users.

## The Core Principles

- **Clarity over Complexity**: Every element should have a clear purpose.
- **Whitespace as a Tool**: Using space to create hierarchy and focus.
- **Intentionality**: Every design choice should be deliberate.

## The Benefits of a Minimalist Approach

1. **Faster Loading Times**: Fewer elements mean less data to load.
2. **Improved Accessibility**: Clearer interfaces are easier for everyone to use.
3. **Longer Lifespan**: Minimalist designs tend to age better than those following fleeting trends.

Minimalism is a journey, not a destination. It requires constant refinement and a deep understanding of the user's needs.
    `,
  },
  {
    id: "typography",
    title: "Hunting Lateral Movement with Splunk and Atomic Red Teams",
    date: "Jan 05, 2026",
    readTime: "4 min read",
    image: "https://mustafabohra.is-a.dev/assets/img/to_post/graph-3.png",
    content: `
# The Nuance of Typography

Typography is the voice of your brand. It conveys personality, tone, and authority without saying a single word.

## The Power of Choice

Selecting the right typeface is a critical decision. It can make or break a design.

- **Serif vs. Sans Serif**: Understanding the historical and psychological implications of each.
- **Hierarchy and Scale**: Using size and weight to guide the reader's eye.
- **Legibility and Readability**: Ensuring that your content is easy to consume.

## The Future of Typography

Variable fonts and responsive typography are changing the way we think about type on the web. We now have more control than ever before, allowing us to create truly unique and engaging reading experiences.

Typography is an art form, and like any art form, it requires practice and a keen eye for detail.
    `,
  },
  {
    id: "scalable-systems",
    title: "Building a Phishing Simulation Framework from Scratch",
    date: "Dec 18, 2025",
    readTime: "12 min read",
    image: "https://mustafabohra.is-a.dev/assets/img/to_post/phision.png",
    content: `
    
# Building Scalable Systems

Scalability is not just about handling more traffic; it's about building systems that can grow and evolve over time.

## The Foundation of Scalability

- **Modular Design**: Breaking down complex systems into smaller, manageable components.
- **Automation**: Reducing manual effort and minimizing errors.
- **Monitoring and Observability**: Understanding how your system is performing in real-time.

## The Challenges of Scaling

1. **Technical Debt**: Balancing speed of development with long-term maintainability.
2. **Cultural Shifts**: Ensuring that your team is aligned and working towards the same goals.
3. **Security and Compliance**: Protecting your data and meeting regulatory requirements.

Building scalable systems is a continuous process. It requires a long-term vision and a commitment to excellence.

    `,
  },
];

export default function Journal() {
  const [selectedPost, setSelectedPost] = useState<JournalEntry | null>(null);

  return (
    <section className="bg-bg py-16 md:py-32">
      <div className="max-w-[1200px] mx-auto px-6 md:px-10 lg:px-16">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: [0.25, 0.1, 0.25, 1] }}
          viewport={{ once: true, margin: "-100px" }}
          className="flex flex-col md:flex-row md:items-end justify-between gap-8 mb-16"
        >
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-px bg-stroke" />
              <span className="text-xs text-muted uppercase tracking-[0.3em]">Selected work</span>
            </div>
            <h2 className="text-4xl md:text-6xl font-display italic text-text-primary mb-4">
              Case <span className="text-muted">studies</span>
            </h2>
            <p className="text-muted max-w-sm">
              A selection of projects I've worked on, from concept to launch.
            </p>
          </div>

          <button className="group relative hidden md:inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm text-text-primary transition-all">
            <div className="absolute inset-0 rounded-full accent-gradient opacity-0 group-hover:opacity-100 -m-[1px] transition-opacity" />
            <div className="relative z-10 flex items-center gap-2 bg-bg rounded-full px-6 py-3 -mx-6 -my-3">
              View all articles <ArrowRight className="w-4 h-4" />
            </div>
          </button>
        </motion.div>

        <div className="flex flex-col gap-4">
          {entries.map((entry, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              onClick={() => setSelectedPost(entry)}
              className="group flex flex-col sm:flex-row items-start sm:items-center gap-6 p-4 sm:p-6 bg-surface/30 hover:bg-surface border border-stroke rounded-[32px] sm:rounded-full transition-all duration-500 cursor-pointer"
            >
              <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full overflow-hidden flex-shrink-0 border border-stroke">
                <img
                  src={entry.image}
                  alt={entry.title}
                  className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                  referrerPolicy="no-referrer"
                />
              </div>

              <div className="flex-1">
                <h3 className="text-lg md:text-xl font-medium text-text-primary group-hover:text-muted transition-colors">
                  {entry.title}
                </h3>
                <div className="flex items-center gap-4 mt-1">
                  <span className="text-xs text-muted">{entry.date}</span>
                  <div className="w-1 h-1 rounded-full bg-stroke" />
                  <span className="text-xs text-muted">{entry.readTime}</span>
                </div>
              </div>

              <div className="hidden sm:flex w-12 h-12 rounded-full border border-stroke items-center justify-center group-hover:bg-text-primary group-hover:border-text-primary transition-all duration-500">
                <ArrowRight className="w-5 h-5 text-muted group-hover:text-bg transition-colors" />
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      <AnimatePresence>
        {selectedPost && (
          <JournalPost
            post={selectedPost}
            onClose={() => setSelectedPost(null)}
          />
        )}
      </AnimatePresence>
    </section>
  );
}

