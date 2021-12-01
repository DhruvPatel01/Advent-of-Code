(defn day1 [ans a b c s]
  (let [line (read-line)]
    (if (nil? line)
      ans
      (let [d (Integer/parseInt line)
            s' (- (+ s d) a)]
        (if (> s' s)
          (recur (+ ans 1) b c d s')
          (recur ans b c d s')
          )
        ))
    )
  )

(println (day1 -3 0 0 0 0))
