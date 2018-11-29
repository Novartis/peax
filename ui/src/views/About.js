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
            <span className="name">
              Fritz Lekschas
              <span className="badge role">Concept, Design, Engineering</span>
            </span>
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
          </li>
          <li className="flex-c flex-v">
            <span className="name">
              Eric Ma
              <span className="badge role">Engineering</span>
            </span>
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
        </ol>

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
              by Bernar Novalyi
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
              by Bhuvan
            </p>
          </li>
        </ul>
      </div>
    </Content>
    <Footer />
  </ContentWrapper>
);

export default About;
