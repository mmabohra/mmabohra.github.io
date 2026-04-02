import { motion } from "framer-motion";

const stats = [
  { value: "2+", label: "Years Experience" },
  { value: "25+", label: "Projects Done" },
  { value: "200%", label: "Satisfied Clients" },
];

export default function Stats() {
  return (
    <section className="bg-bg py-24 md:py-40 border-y border-stroke">
      <div className="max-w-[1200px] mx-auto px-6 md:px-10 lg:px-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 md:gap-24">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="text-center md:text-left"
            >
              <div className="text-6xl md:text-8xl font-display italic text-text-primary mb-4">
                {stat.value}
              </div>
              <div className="text-xs text-muted uppercase tracking-[0.3em]">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
