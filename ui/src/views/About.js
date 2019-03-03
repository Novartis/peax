import React from 'react';

// Components
import Content from '../components/Content';
import ContentWrapper from '../components/ContentWrapper';
import Footer from '../components/Footer';
import Icon from '../components/Icon';

// Stylesheets
import './About.scss';

const About = () => (
  <ContentWrapper name="about">
    <Content name="about">
      <div className="wrap p-b-2">
        <h3 id="abstract" className="iconized underlined anchored">
          <a href="#abstract" className="hidden-anchor">
            <Icon iconId="link" />
          </a>
          <Icon iconId="text" />
          <span>Summary</span>
        </h3>
        <p className="abstract">
          Epigenomic data expresses a rich body of diverse patterns, but
          extracting patterns genome wide is limited, as specialized algorithms
          are required or the expressiveness is low. Peax is a tool for
          interactive concept learning and exploration of epigenomic patterns
          based on unsupervised featurization with autorencoders. Genomic
          regions are manually labeled for actively learning feature weights to
          build a custom classifiers based on a researchers notion of
          interstingness.
        </p>

        <h3 id="abstract" className="iconized underlined anchored">
          <a href="#abstract" className="hidden-anchor">
            <Icon iconId="link" />
          </a>
          <Icon iconId="person" />
          <span>Authors</span>
        </h3>

        <ol className="no-list-style authors">
          <li className="flex-c flex-v">
            <span className="name">Fritz Lekschas</span>
            <span className="affiliation">
              Harvard John A. Paulson School of Engineering and Applied Sciences
            </span>
            <ul className="no-list-style flex-c out-links">
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://lekschas.de"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="globe" />
                  lekschas.de
                </a>
              </li>
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://twitter.com/flekschas"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="gezwitscher" />
                  flekschas
                </a>
              </li>
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://github.com/flekschas"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="github" />
                  flekschas
                </a>
              </li>
            </ul>
          </li>
          <li className="flex-c flex-v">
            <span className="name">
              Brant Peterson
              <span className="badge role">Concept</span>
            </span>
            <span className="affiliation">
              Novartis Institutes for BioMedical Research
            </span>
            <ul className="no-list-style flex-c out-links">
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://twitter.com/fixedbydrift"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="gezwitscher" />
                  fixedbydrift
                </a>
              </li>
            </ul>
          </li>
          <li className="flex-c flex-v">
            <span className="name">Daniel Haehn</span>
            <span className="affiliation">
              Harvard John A. Paulson School of Engineering and Applied Sciences
            </span>
            <ul className="no-list-style flex-c out-links">
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://danielhaehn.com"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="globe" />
                  danielhaehn.com
                </a>
              </li>
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://twitter.com/danielhaehn"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="gezwitscher" />
                  danielhaehn
                </a>
              </li>
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://github.com/haehn"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="github" />
                  haehn
                </a>
              </li>
            </ul>
          </li>
          <li className="flex-c flex-v">
            <span className="name">Eric Ma</span>
            <span className="affiliation">
              Novartis Institutes for BioMedical Research
            </span>
            <ul className="no-list-style flex-c out-links">
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="http://ericmjl.com"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="globe" />
                  ericmjl.com
                </a>
              </li>
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://twitter.com/ericmjl"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="gezwitscher" />
                  ericmjl
                </a>
              </li>
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://github.com/ericmjl"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="github" />
                  ericmjl
                </a>
              </li>
            </ul>
          </li>
          <li className="flex-c flex-v">
            <span className="name">Nils Gehlenborg</span>
            <span className="affiliation">Harvard Medical School</span>
            <ul className="no-list-style flex-c out-links">
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="http://gehlenborglab.org"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="globe" />
                  gehlenborglab.org
                </a>
              </li>
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://twitter.com/ngehlenborg"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="gezwitscher" />
                  ngehlenborg
                </a>
              </li>
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://github.com/ngehlenborg"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="github" />
                  ngehlenborg
                </a>
              </li>
            </ul>
          </li>
          <li className="flex-c flex-v">
            <span className="name">Hanspeter Pfister</span>
            <span className="affiliation">
              Harvard John A. Paulson School of Engineering and Applied Sciences
            </span>
            <ul className="no-list-style flex-c out-links">
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://vcg.seas.harvard.edu/people"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="globe" />
                  vcg.seas.harvard.edu
                </a>
              </li>
              <li>
                <a
                  className="flex-c flex-a-c"
                  href="https://twitter.com/hpfister"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon iconId="gezwitscher" />
                  hpfister
                </a>
              </li>
            </ul>
          </li>
        </ol>

        <h3 id="source-code" className="iconized underlined anchored">
          <a href="#source-code" className="hidden-anchor">
            <Icon iconId="link" />
          </a>
          <Icon iconId="code" />
          Source Code
        </h3>

        <ul className="no-list-style large-spacing iconized">
          <li className="iconized">
            <Icon iconId="github" />
            <span className="m-r-0-5">Repo:</span>
            <a
              href="https://github.com/novartis/peax"
              target="_blank"
              rel="noopener noreferrer"
            >
              github.com/novartis/peax
            </a>
          </li>
        </ul>

        <p>Peax uses and adopts the following open source component:</p>

        <ul className="no-list-style large-spacing iconized">
          <li className="iconized">
            <Icon iconId="github" />
            <span className="m-r-0-5">Genome viewer:</span>
            <a
              href="https://github.com/higlass/higlass"
              target="_blank"
              rel="noopener noreferrer"
            >
              github.com/higlass/higlass
            </a>
          </li>
          <li className="iconized">
            <Icon iconId="github" />
            <span className="m-r-0-5">UI architecture:</span>
            <a
              href="https://github.com/higlass/higlass-app"
              target="_blank"
              rel="noopener noreferrer"
            >
              github.com/higlass/higlass-app
            </a>
          </li>
          <li className="iconized">
            <Icon iconId="github" />
            <span className="m-r-0-5">Server:</span>
            <a
              href="https://github.com/higlass/hgflask"
              target="_blank"
              rel="noopener noreferrer"
            >
              github.com/higlass/hgflask
            </a>
          </li>
        </ul>

        <h3 id="design" className="iconized underlined anchored">
          <a href="#design" className="hidden-anchor">
            <Icon iconId="link" />
          </a>
          <Icon iconId="pen-ruler" />
          Design
        </h3>

        <p>
          The website and logo (<Icon iconId="logo" inline />) are designed by{' '}
          <a
            href="https://lekschas.de"
            target="_blank"
            rel="noopener noreferrer"
          >
            Fritz Lekschas
          </a>
          .
        </p>

        <h3 id="copyright" className="iconized underlined anchored">
          <a href="#copyright" className="hidden-anchor">
            <Icon iconId="link" />
          </a>
          <Icon iconId="info-circle" />
          Icons
        </h3>

        <p>
          The following sets of beautiful icons have been slightly adjusted by
          Fritz Lekschas and are used across the application. Hugs thanks to the
          authors for their fantastic work!
        </p>

        <ul className="no-list-style large-spacing iconized">
          <li className="flex-c iconized">
            <Icon iconId="code" />
            <p className="nm">
              <a
                href="https://thenounproject.com/term/code/821469/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Code
              </a>
              <span> by Bernar Novalyi</span>
            </p>
          </li>
          <li className="flex-c iconized">
            <Icon iconId="launch" />
            <p className="nm">
              <a
                href="https://thenounproject.com/icon/1372884/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Launch
              </a>
              <span> by Bhuvan</span>
            </p>
          </li>
        </ul>
      </div>
    </Content>
    <Footer />
  </ContentWrapper>
);

export default About;
