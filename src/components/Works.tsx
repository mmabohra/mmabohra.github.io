import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";

const projects = [
  {
    title: "Adversarial Detection",
    image: "https://i1.rgstatic.net/publication/335321361_On_Detection_of_False_Data_in_Cooperative_DC_Microgrids-A_Discordant_Element_Approach/links/5d5e7282299bf1b97cfd6071/largepreview.png",
    span: "md:col-span-7",
    aspect: "aspect-[16/10] md:aspect-auto md:h-[400px]",
  },
  {
    title: "Triage Automation",
    image: "https://www.runtime.news/content/images/size/w1200/2025/03/Screenshot-2025-03-27-at-3.00.54-PM.png",
    span: "md:col-span-5",
    aspect: "aspect-[4/5] md:aspect-auto md:h-[400px]",
  },
  {
    title: "Detection Engineering",
    image: "https://mustafabohra.is-a.dev/assets/img/to_post/graph-3.png",
    span: "md:col-span-5",
    aspect: "aspect-[4/5] md:aspect-auto md:h-[400px]",
  },
  {
    title: "Framework Simulation",
    image: "https://i.pinimg.com/736x/bd/da/a8/bddaa800e3b5c725883adf34b619d76a.jpg",
    span: "md:col-span-7",
    aspect: "aspect-[16/10] md:aspect-auto md:h-[400px]",
  },
];

export default function Works() {
  return (
    <section id="work" className="bg-bg py-12 md:py-24">
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
              <span className="text-xs text-muted uppercase tracking-[0.3em]">Selected Work</span>
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
              View all work <ArrowRight className="w-4 h-4" />
            </div>
          </button>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-5 md:gap-6">
          {projects.map((project, index) => (
            <ProjectCard key={index} {...project} />
          ))}
        </div>
      </div>
    </section>
  );
}

function ProjectCard({ title, image, span, aspect }: typeof projects[0] & { key?: any }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, delay: 0.1 }}
      viewport={{ once: true }}
      className={`${span} group relative bg-surface border border-stroke rounded-3xl overflow-hidden cursor-pointer`}
    >
      <div className={`${aspect} w-full relative overflow-hidden`}>
        <img
          src={image}
          alt={title}
          className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105"
          referrerPolicy="no-referrer"
        />

        {/* Halftone Overlay */}
        <div
          className="absolute inset-0 opacity-20 mix-blend-multiply pointer-events-none"
          style={{ backgroundImage: "radial-gradient(circle, #000 1px, transparent 1px)", backgroundSize: "4px 4px" }}
        />

        {/* Hover Overlay */}
        <div className="absolute inset-0 bg-bg/70 backdrop-blur-lg opacity-0 group-hover:opacity-100 transition-all duration-500 flex items-center justify-center p-6">
          <div className="group/label relative px-6 py-3 rounded-full bg-white text-bg overflow-hidden transition-transform duration-500 translate-y-4 group-hover:translate-y-0">
            <div className="absolute inset-0 accent-gradient animate-gradient-shift opacity-0 group-hover/label:opacity-100 transition-opacity" />
            <span className="relative z-10 text-sm font-medium">
              View — <span className="font-display italic">{title}</span>
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
